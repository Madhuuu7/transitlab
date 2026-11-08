/**
 * Decode the browser bundle written by `transitlab.gtfs.webbundle`.
 *
 * The encoding exists so the feed is 1.9 MB instead of 7.9 MB over the wire;
 * see the Python module for why those particular two tricks work on this
 * feed. This file is the only place that knows about them. By the time a
 * {@link Timetable} exists the full int32 stop-time tables have been rebuilt,
 * so the router runs the same arithmetic the server does and no part of it
 * has to care how the bytes arrived.
 */

import { Timetable, type Stop } from "./timetable.ts";

export const BUNDLE_VERSION = 2;

export interface ArraySpec {
  name: string;
  dtype: "i32" | "u16" | "f64";
  length: number;
  offset: number;
  bytes: number;
}

export interface BundleManifest {
  version: number;
  feed: string;
  encoding: "u16-delta" | "i32-absolute";
  dwell: boolean;
  counts: {
    stops: number;
    patterns: number;
    trips: number;
    stop_times: number;
    footpaths: number;
  };
  arrays: ArraySpec[];
  /**
   * Compressed size of each file, recorded by the build.
   *
   * The console reports its own download from this rather than from what it
   * observed, because a host that sets `Content-Encoding: gzip` on a `.gz`
   * file has the browser inflate it before any code can count it -- the
   * figure would then describe the host instead of the feed.
   */
  download?: Record<string, number>;
}

export interface BundleStrings {
  stop_ids: string[];
  stop_names: string[];
  pattern_route_id: string[];
  pattern_route_name: string[];
  trip_ids: string[];
}

export class BundleError extends Error {}

function views(manifest: BundleManifest, binary: ArrayBuffer) {
  const out = new Map<string, Int32Array | Uint16Array | Float64Array>();
  for (const spec of manifest.arrays) {
    const itemsize = spec.dtype === "i32" ? 4 : spec.dtype === "u16" ? 2 : 8;
    if (spec.offset + spec.length * itemsize > binary.byteLength) {
      throw new BundleError(`${spec.name} runs past the end of the bundle`);
    }
    if (spec.offset % itemsize !== 0) {
      // A typed array over a buffer refuses a misaligned offset, so the
      // encoder pads every array to 8 bytes. If this ever fires, the bundle
      // was not written by webbundle.py.
      throw new BundleError(`${spec.name} is misaligned at byte ${spec.offset}`);
    }
    const ctor =
      spec.dtype === "i32" ? Int32Array : spec.dtype === "u16" ? Uint16Array : Float64Array;
    out.set(spec.name, new ctor(binary, spec.offset, spec.length));
  }
  return out;
}

function required<T>(map: Map<string, unknown>, name: string): T {
  const array = map.get(name);
  if (!array) throw new BundleError(`bundle is missing ${name}`);
  return array as T;
}

/**
 * One base departure per stop-time, expanded from one per trip.
 *
 * Mirrors `_expand_base` in the Python encoder; the two must agree exactly or
 * every time in the feed is wrong by a trip's start.
 */
function expandBase(
  tripFirst: Int32Array,
  patternStopOffset: Int32Array,
  patternTripOffset: Int32Array,
  stopTimeOffset: Int32Array,
  nStopTimes: number,
): Int32Array {
  const out = new Int32Array(nStopTimes);
  const nPatterns = patternStopOffset.length - 1;
  for (let pattern = 0; pattern < nPatterns; pattern += 1) {
    const length = patternStopOffset[pattern + 1] - patternStopOffset[pattern];
    const trips = patternTripOffset[pattern + 1] - patternTripOffset[pattern];
    const base = stopTimeOffset[pattern];
    const lo = patternTripOffset[pattern];
    for (let trip = 0; trip < trips; trip += 1) {
      const value = tripFirst[lo + trip];
      const start = base + trip * length;
      out.fill(value, start, start + length);
    }
  }
  return out;
}

export function decodeBundle(
  manifest: BundleManifest,
  binary: ArrayBuffer,
  strings: BundleStrings,
): Timetable {
  if (manifest.version !== BUNDLE_VERSION) {
    throw new BundleError(
      `bundle is version ${manifest.version}, this console reads ${BUNDLE_VERSION}`,
    );
  }

  const raw = views(manifest, binary);
  const patternStopOffset = required<Int32Array>(raw, "pattern_stop_offset");
  const patternTripOffset = required<Int32Array>(raw, "pattern_trip_offset");
  const stopTimeOffset = required<Int32Array>(raw, "stop_time_offset");

  let departures: Int32Array;
  let arrivals: Int32Array;

  if (manifest.encoding === "u16-delta") {
    const base = expandBase(
      required<Int32Array>(raw, "trip_first_departure"),
      patternStopOffset,
      patternTripOffset,
      stopTimeOffset,
      manifest.counts.stop_times,
    );
    const departureOffset = required<Uint16Array>(raw, "departure_offset");
    departures = new Int32Array(base.length);
    for (let i = 0; i < departures.length; i += 1) {
      departures[i] = base[i] + departureOffset[i];
    }
    if (manifest.dwell) {
      const arrivalOffset = required<Uint16Array>(raw, "arrival_offset");
      arrivals = new Int32Array(base.length);
      for (let i = 0; i < arrivals.length; i += 1) {
        arrivals[i] = base[i] + arrivalOffset[i];
      }
    } else {
      // The feed publishes one time per stop, so these are the same numbers.
      // Sharing the array is not an optimisation to be clever about -- it is
      // what the manifest says is true of this feed.
      arrivals = departures;
    }
  } else if (manifest.encoding === "i32-absolute") {
    departures = required<Int32Array>(raw, "departures");
    arrivals = manifest.dwell ? required<Int32Array>(raw, "arrivals") : departures;
  } else {
    throw new BundleError(`unknown time encoding ${manifest.encoding}`);
  }

  const coords = required<Float64Array>(raw, "coords");
  const { stop_ids: ids, stop_names: names } = strings;
  if (ids.length !== names.length || ids.length * 2 !== coords.length) {
    throw new BundleError("bundle has inconsistent stop metadata");
  }
  const stops: Stop[] = ids.map((stopId, index) => ({
    stopId,
    name: names[index],
    lat: coords[index * 2],
    lon: coords[index * 2 + 1],
  }));

  return new Timetable({
    stops,
    coords,
    patternRouteId: strings.pattern_route_id,
    patternRouteName: strings.pattern_route_name,
    patternStopOffset,
    patternStops: required<Int32Array>(raw, "pattern_stops"),
    patternTripOffset,
    tripIds: strings.trip_ids,
    stopTimeOffset,
    arrivals,
    departures,
    stopPatternOffset: required<Int32Array>(raw, "stop_pattern_offset"),
    stopPatternPattern: required<Int32Array>(raw, "stop_pattern_pattern"),
    stopPatternPosition: required<Int32Array>(raw, "stop_pattern_position"),
    footpathOffset: required<Int32Array>(raw, "footpath_offset"),
    footpathTarget: required<Int32Array>(raw, "footpath_target"),
    footpathTime: required<Int32Array>(raw, "footpath_time"),
  });
}
