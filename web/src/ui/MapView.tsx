/**
 * The map.
 *
 * Four layers, in the order transit cartography puts them: the whole network
 * as faint dots, the isochrone over it, then the journey's walking segments
 * and ride segments with a dark casing under each, then the endpoints. The
 * casing is not styling -- a 3 px line over a dark basemap disappears against
 * a road of similar value, and a darker stroke beneath it is what keeps it
 * readable. Every printed transit map does the same thing.
 *
 * Ride legs are drawn straight between the stops where you board and alight.
 * The feed ships no `shapes.txt`, so the road geometry is not available at
 * any price, and a curve invented to look like a road would be a drawing of
 * data that does not exist. The diagram in the rail carries the sequence; the
 * map carries the geography.
 */

import { useEffect, useRef, useState } from "react";
// maplibre-gl 6 publishes named exports and no default, and no longer brings
// the GeoJSON namespace in globally, so the geometry types are imported too.
import {
  LngLatBounds,
  Map as MapLibreMap,
  NavigationControl,
  Popup,
  setWorkerUrl,
  type GeoJSONSource,
} from "maplibre-gl";
import type { Feature, FeatureCollection, LineString, Point } from "geojson";

// maplibre 6 loads its tile worker from a separate module and resolves the
// URL itself, which a bundler cannot see -- the map then fails at runtime
// with "Worker failed to load" and renders nothing, while the build reports
// success. `?worker&url` has vite bundle that module as a worker, resolve
// its own imports, and hand back the emitted asset's URL, which stays
// correct under a project-site base path.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

setWorkerUrl(workerUrl);

import type { JourneyOut, ReachedStop, StopOut } from "../engine/api.ts";
import type { Stop, Timetable } from "../engine/timetable.ts";
import { RAMP, rampColor, rampExpression } from "./ramp.ts";

const BENGALURU: [number, number] = [77.5946, 12.9716];

interface Props {
  stops: Stop[];
  timetable: Timetable;
  journey: JourneyOut | null;
  origin: StopOut | null;
  destination: StopOut | null;
  reached: ReachedStop[] | null;
}

/**
 * Every corridor a bus runs along, as one multi-line.
 *
 * This is the basemap. Each pattern is a sequence of stops, so each
 * consecutive pair is a segment some bus actually drives; collapsing the
 * 228,169 pairs across 7,838 patterns down to the unique ones leaves the
 * shape of the city's bus network, drawn entirely from the feed already in
 * memory. No tiles, no key, no second source of truth to go stale.
 *
 * Straight segments again, for the same reason the journey legs are: the
 * feed has no road geometry. At city scale, with stops roughly every 250 m,
 * the chords and the roads are the same lines to the eye.
 */
function networkEdges(timetable: Timetable): FeatureCollection {
  const seen = new Set<number>();
  const coordinates: Array<[number, number][]> = [];
  const stops = timetable.nStops;

  for (let pattern = 0; pattern < timetable.nPatterns; pattern += 1) {
    const sequence = timetable.stopsOf(pattern);
    for (let at = 1; at < sequence.length; at += 1) {
      const a = sequence[at - 1];
      const b = sequence[at];
      if (a === b) continue;
      const key = a < b ? a * stops + b : b * stops + a;
      if (seen.has(key)) continue;
      seen.add(key);
      coordinates.push([
        [timetable.lonOf(a), timetable.latOf(a)],
        [timetable.lonOf(b), timetable.latOf(b)],
      ]);
    }
  }

  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: { segments: coordinates.length },
        geometry: { type: "MultiLineString", coordinates },
      },
    ],
  };
}

function emptyCollection(): FeatureCollection {
  return { type: "FeatureCollection", features: [] };
}

/** Walk and ride legs as line features, coloured by elapsed time. */
function journeyLines(
  journey: JourneyOut | null,
  lookup: Map<string, Stop>,
  departure: number,
): FeatureCollection {
  if (!journey) return emptyCollection();
  const features: Feature[] = [];
  let elapsed = 0;
  for (const leg of journey.legs) {
    const from = lookup.get(leg.from_stop);
    const to = lookup.get(leg.to_stop);
    elapsed += leg.seconds;
    if (!from || !to) continue;
    features.push({
      type: "Feature",
      properties: {
        kind: leg.kind,
        color: rampColor(Math.round(elapsed / 60)),
        route: leg.route_id ?? "",
      },
      geometry: {
        type: "LineString",
        coordinates: [
          [from.lon, from.lat],
          [to.lon, to.lat],
        ],
      },
    });
  }
  void departure;
  return { type: "FeatureCollection", features };
}

function journeyPoints(
  journey: JourneyOut | null,
  lookup: Map<string, Stop>,
): FeatureCollection {
  if (!journey) return emptyCollection();
  const features: Feature[] = [];
  const seen = new Set<string>();
  journey.legs.forEach((leg, at) => {
    const ends: Array<[string, boolean]> = [
      [leg.from_stop, at === 0],
      [leg.to_stop, at === journey.legs.length - 1],
    ];
    for (const [stopId, terminal] of ends) {
      if (seen.has(stopId)) continue;
      seen.add(stopId);
      const stop = lookup.get(stopId);
      if (!stop) continue;
      features.push({
        type: "Feature",
        properties: { name: stop.name, code: stop.stopId, terminal: terminal ? 1 : 0 },
        geometry: { type: "Point", coordinates: [stop.lon, stop.lat] },
      });
    }
  });
  return { type: "FeatureCollection", features };
}

function reachedPoints(reached: ReachedStop[] | null): FeatureCollection {
  if (!reached) return emptyCollection();
  return {
    type: "FeatureCollection",
    features: reached.map((stop) => ({
      type: "Feature",
      properties: { minutes: stop.minutes, name: stop.name, arrival: stop.arrival },
      geometry: { type: "Point", coordinates: [stop.lon, stop.lat] },
    })),
  };
}

export function MapView({
  stops,
  timetable,
  journey,
  origin: originStop,
  destination,
  reached,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  // State, not a ref. The feed decodes faster than the map finishes setting
  // up, so the first journey is already computed when the data effects first
  // run; with a ref they would bail out and never re-run, and the map would
  // sit empty beside a rail showing a complete journey.
  const [ready, setReady] = useState(false);
  const lookup = useRef(new Map<string, Stop>());

  useEffect(() => {
    lookup.current = new Map(stops.map((stop) => [stop.stopId, stop]));
  }, [stops]);

  // Set up once. MapLibre owns its own canvas, so React must not re-create it.
  useEffect(() => {
    if (!host.current || map.current) return;
    const instance = new MapLibreMap({
      container: host.current,
      center: BENGALURU,
      zoom: 10.6,
      attributionControl: {
        compact: true,
        customAttribution:
          'Schedules and stop positions © <a href="https://github.com/Vonter/bmtc-gtfs">' +
          "BMTC via Vonter</a>, ODbL 1.0",
      },
      style: {
        version: 8,
        sources: {},
        // No basemap. Not a saving -- a decision, and the one place this
        // console departs furthest from how a map is usually built.
        //
        // The practical reason is that there is no keyless dark raster
        // basemap left: CARTO's now answers with an API-KEY-REQUIRED tile,
        // and OSM's own tiles are light and ask not to be used this way. The
        // better reason is that a borrowed basemap was never carrying its
        // weight here. 9,948 bus stops at 250 m spacing already draw
        // Bengaluru's road network, because that is where buses stop. The
        // city on screen is the feed, which is the only thing on this page
        // whose accuracy is being claimed.
        layers: [{ id: "ground", type: "background", paint: { "background-color": "#0c1114" } }],
      },
    });
    instance.addControl(new NavigationControl({ showCompass: false }), "top-right");
    instance.on("load", () => {
      for (const id of ["corridors", "network", "reached", "legs", "ends"]) {
        instance.addSource(id, { type: "geojson", data: emptyCollection() });
      }

      instance.addLayer({
        id: "corridors",
        type: "line",
        source: "corridors",
        paint: {
          // Dim enough to read as ground rather than as an answer, but it is
          // the only thing giving the city a shape, so it carries more weight
          // than the stop dots over it.
          "line-color": "#24383f",
          "line-width": ["interpolate", ["linear"], ["zoom"], 9, 0.5, 14, 1.6],
          "line-opacity": 0.9,
        },
      });

      instance.addLayer({
        id: "network",
        type: "circle",
        source: "network",
        paint: {
          // Smaller than anything drawn over it: context, not the answer.
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 0.7, 14, 2],
          "circle-color": "#4e666d",
          "circle-opacity": 0.75,
        },
      });

      instance.addLayer({
        id: "reached",
        type: "circle",
        source: "reached",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 3.1, 14, 6.5],
          "circle-color": rampExpression() as never,
          "circle-opacity": 0.82,
          "circle-stroke-width": 0.4,
          "circle-stroke-color": "#05080a",
        },
      });

      instance.addLayer({
        id: "legs-casing",
        type: "line",
        source: "legs",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#05080a", "line-width": 7, "line-opacity": 0.9 },
      });

      instance.addLayer({
        id: "legs-walk",
        type: "line",
        source: "legs",
        filter: ["==", ["get", "kind"], "walk"],
        layout: { "line-cap": "butt" },
        paint: { "line-color": "#9fb0ab", "line-width": 2.4, "line-dasharray": [1, 1.8] },
      });

      instance.addLayer({
        id: "legs-ride",
        type: "line",
        source: "legs",
        filter: ["==", ["get", "kind"], "ride"],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": ["get", "color"], "line-width": 3.6 },
      });

      instance.addLayer({
        id: "ends",
        type: "circle",
        source: "ends",
        paint: {
          "circle-radius": ["case", ["==", ["get", "terminal"], 1], 7, 5],
          "circle-color": "#05080a",
          "circle-stroke-width": ["case", ["==", ["get", "terminal"], 1], 3, 2.4],
          "circle-stroke-color": ["case", ["==", ["get", "terminal"], 1], "#e8f0ee", "#5fd6a0"],
        },
      });

      const popup = new Popup({
        closeButton: false,
        offset: 11,
        maxWidth: "260px",
      });
      for (const layer of ["reached", "ends", "network"]) {
        instance.on("mouseenter", layer, (event) => {
          instance.getCanvas().style.cursor = "pointer";
          const feature = event.features?.[0];
          if (!feature) return;
          const props = feature.properties ?? {};
          const minutes = props.minutes;
          // textContent, never innerHTML: these names come out of a
          // third-party feed and go straight onto the page.
          const node = document.createElement("div");
          const name = document.createElement("strong");
          name.textContent = String(props.name ?? "");
          node.appendChild(name);
          if (minutes !== undefined && minutes !== null) {
            const detail = document.createElement("em");
            detail.textContent = `${minutes} min · arrive ${props.arrival ?? ""}`;
            node.appendChild(detail);
          } else if (props.code) {
            const detail = document.createElement("em");
            detail.textContent = String(props.code);
            node.appendChild(detail);
          }
          popup
            .setLngLat((feature.geometry as Point).coordinates as [number, number])
            .setDOMContent(node)
            .addTo(instance);
        });
        instance.on("mouseleave", layer, () => {
          instance.getCanvas().style.cursor = "";
          popup.remove();
        });
      }

      // Everything else is filled by the data effects below, which now run
      // again because this is state.
      setReady(true);
    });

    map.current = instance;
    return () => {
      instance.remove();
      map.current = null;
      setReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The city: every corridor a bus runs along, and every stop.
  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready) return;
    (instance.getSource("corridors") as GeoJSONSource | undefined)?.setData(
      networkEdges(timetable),
    );
    (instance.getSource("network") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection",
      features: stops.map((stop) => ({
        type: "Feature",
        properties: { name: stop.name, code: stop.stopId },
        geometry: { type: "Point", coordinates: [stop.lon, stop.lat] },
      })),
    });
  }, [stops, timetable, ready]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready) return;
    const lines = journeyLines(journey, lookup.current, 0);
    (instance.getSource("legs") as GeoJSONSource | undefined)?.setData(lines);
    (instance.getSource("ends") as GeoJSONSource | undefined)?.setData(
      journeyPoints(journey, lookup.current),
    );

    if (!journey || lines.features.length === 0) return;
    const bounds = new LngLatBounds();
    for (const feature of lines.features) {
      for (const point of (feature.geometry as LineString).coordinates) {
        bounds.extend(point as [number, number]);
      }
    }
    instance.fitBounds(bounds, { padding: 84, maxZoom: 14.5, duration: 620 });
  }, [journey, ready]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready) return;
    (instance.getSource("reached") as GeoJSONSource | undefined)?.setData(
      reachedPoints(reached),
    );
    if (!reached || reached.length === 0) return;
    // Framed on the bulk of the set, not its extremes. A few long-distance
    // routes reach stops 40 km out, and fitting those puts the whole city in
    // a thumbnail-sized blob in one corner -- technically the right bounds
    // and the wrong picture. Everything stays drawn; only the opening frame
    // ignores the farthest two percent, and panning out still shows them.
    // The real origin when we have it; reached is sorted by minutes, so its
    // first entry is the origin itself in every ordinary case.
    const origin = originStop ?? reached[0];
    const spread = reached
      .map((stop) => Math.hypot(stop.lon - origin.lon, stop.lat - origin.lat))
      .sort((a, b) => a - b);
    const cutoff = spread[Math.floor(spread.length * 0.98)] || Infinity;
    const bounds = new LngLatBounds();
    bounds.extend([origin.lon, origin.lat]);
    for (const stop of reached) {
      if (Math.hypot(stop.lon - origin.lon, stop.lat - origin.lat) <= cutoff) {
        bounds.extend([stop.lon, stop.lat]);
      }
    }
    instance.fitBounds(bounds, { padding: 72, maxZoom: 13, duration: 620 });
  }, [reached, ready]);

  // Keep the endpoints in view when only the query changed.
  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready || journey || reached) return;
    const points = [originStop, destination].filter(Boolean) as StopOut[];
    if (points.length === 0) return;
    const bounds = new LngLatBounds();
    for (const stop of points) bounds.extend([stop.lon, stop.lat]);
    instance.fitBounds(bounds, { padding: 96, maxZoom: 13.5, duration: 420 });
  }, [originStop, destination, journey, reached, ready]);

  return (
    <div className="canvas">
      <div className="map" ref={host} />
      {reached && (
        <div className="legend">
          <p className="legend__title">Minutes from the origin</p>
          <div className="legend__scale">
            {RAMP.map((step) => (
              <div className="legend__step" key={step.label}>
                <div className="legend__swatch" style={{ background: step.color }} />
                <div className="legend__tick">{step.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
