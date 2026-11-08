/**
 * The HTTP API, answered locally.
 *
 * Every function here returns exactly what the corresponding FastAPI endpoint
 * in `transitlab.api.app` returns, down to the field names and the `null`s on
 * a walk leg. That is deliberate and it is the whole testing strategy: the
 * console is written against one response shape, `web/test/parity.test.ts`
 * compares these objects to answers recorded from the running Python server,
 * and a disagreement is a bug rather than a formatting difference to argue
 * about.
 *
 * It also means the console is not locked to routing in the browser. Point it
 * at a server and the same components render the same responses.
 */

import {
  Raptor,
  journeyDuration,
  journeyTransfers,
  journeyWalkSeconds,
  type Journey,
  type Leg,
} from "./raptor.ts";
import { NEVER, formatGtfsTime, parseGtfsTime, toMinutes } from "./times.ts";
import type { Timetable } from "./timetable.ts";

export interface StopOut {
  stop_id: string;
  name: string;
  lat: number;
  lon: number;
}

export interface LegOut {
  kind: "ride" | "walk";
  from_stop: string;
  to_stop: string;
  from_name: string;
  to_name: string;
  departure: string | null;
  arrival: string | null;
  route_id: string | null;
  trip_id: string | null;
  seconds: number;
}

export interface JourneyOut {
  departure: string;
  arrival: string;
  duration_minutes: number;
  transfers: number;
  walk_minutes: number;
  legs: LegOut[];
}

export interface JourneyResponse {
  origin: StopOut;
  destination: StopOut;
  requested_departure: string;
  fastest: JourneyOut | null;
  alternatives: JourneyOut[];
}

export interface ReachedStop {
  stop_id: string;
  name: string;
  lat: number;
  lon: number;
  arrival: string;
  minutes: number;
}

export interface IsochroneResponse {
  origin: StopOut;
  departure: string;
  budget_minutes: number;
  reached: ReachedStop[];
  unreachable: number;
}

export interface FeedInfo {
  feed: string;
  stops: number;
  patterns: number;
  trips: number;
  stop_times: number;
  footpaths: number;
}

export interface JourneyQuery {
  from: string;
  to: string;
  time?: string;
  changeTime?: number;
  maxWalk?: number | null;
}

export interface IsochroneQuery {
  from: string;
  time?: string;
  minutes?: number;
  changeTime?: number;
  maxWalk?: number | null;
}

/** Matches the API's defaults, so a query left alone behaves identically. */
export const DEFAULTS = {
  time: "08:00:00",
  changeTime: 120,
  minutes: 30,
} as const;

/**
 * Accept `HH:MM` as well as `HH:MM:SS`, as the API's `_parse_time` does,
 * keeping hours past 24 legal.
 */
function parseTime(value: string): number {
  return parseGtfsTime(value);
}

export class LocalEngine {
  readonly timetable: Timetable;
  readonly feedName: string;

  constructor(timetable: Timetable, feedName: string) {
    this.timetable = timetable;
    this.feedName = feedName;
  }

  feed(): FeedInfo {
    const tt = this.timetable;
    return {
      feed: this.feedName,
      stops: tt.nStops,
      patterns: tt.nPatterns,
      trips: tt.nTrips,
      stop_times: tt.nStopTimes,
      footpaths: tt.nFootpaths,
    };
  }

  stopOut(index: number): StopOut {
    const stop = this.timetable.stops[index];
    return { stop_id: stop.stopId, name: stop.name, lat: stop.lat, lon: stop.lon };
  }

  /** `/stops`: an optional case-insensitive filter over name or id. */
  stops(query: { q?: string | null; limit?: number } = {}): StopOut[] {
    const needle = query.q ? query.q.toLowerCase() : null;
    const limit = query.limit ?? 500;
    const out: StopOut[] = [];
    const stops = this.timetable.stops;
    for (let index = 0; index < stops.length; index += 1) {
      const stop = stops[index];
      if (
        needle &&
        !stop.name.toLowerCase().includes(needle) &&
        !stop.stopId.toLowerCase().includes(needle)
      ) {
        continue;
      }
      out.push(this.stopOut(index));
      if (out.length >= limit) break;
    }
    return out;
  }

  resolve(stopId: string, label: string): number {
    const index = this.timetable.stopIndex.get(stopId);
    if (index === undefined) throw new Error(`unknown ${label} stop: ${stopId}`);
    return index;
  }

  /**
   * `/journey`: the fastest journey plus one alternative per transfer count.
   *
   * Target pruning is deliberately off, exactly as in the API: the
   * alternatives read other rounds of the same label table, and a pruned run
   * only has correct labels at the destination.
   */
  journey(query: JourneyQuery): JourneyResponse {
    const source = this.resolve(query.from, "origin");
    const target = this.resolve(query.to, "destination");
    const departure = parseTime(query.time ?? DEFAULTS.time);
    const changeTime = query.changeTime ?? DEFAULTS.changeTime;

    const result = new Raptor(this.timetable, { minChangeTime: changeTime }).run(
      source,
      departure,
      { maxWalkSeconds: query.maxWalk ?? null },
    );

    const fastest = result.journey(target);
    const fastestKey = fastest ? legKey(fastest.legs) : null;
    const alternatives = result
      .journeysByTransfers(target)
      .filter((journey) => fastestKey === null || legKey(journey.legs) !== fastestKey)
      .map((journey) => this.journeyOut(journey));

    return {
      origin: this.stopOut(source),
      destination: this.stopOut(target),
      requested_departure: formatGtfsTime(departure),
      fastest: fastest ? this.journeyOut(fastest) : null,
      alternatives,
    };
  }

  /** `/isochrone`: every stop reachable inside a time budget. */
  isochrone(query: IsochroneQuery): IsochroneResponse {
    const source = this.resolve(query.from, "origin");
    const departure = parseTime(query.time ?? DEFAULTS.time);
    const minutes = query.minutes ?? DEFAULTS.minutes;
    const changeTime = query.changeTime ?? DEFAULTS.changeTime;

    const result = new Raptor(this.timetable, { minChangeTime: changeTime }).run(
      source,
      departure,
      { maxWalkSeconds: query.maxWalk ?? null },
    );

    const reached: ReachedStop[] = result.reachable(minutes * 60).map((index) => {
      const arrival = result.best[index];
      const stop = this.timetable.stops[index];
      return {
        stop_id: stop.stopId,
        name: stop.name,
        lat: stop.lat,
        lon: stop.lon,
        arrival: formatGtfsTime(arrival),
        minutes: toMinutes(arrival - departure),
      };
    });
    // Stable, like Python's sort: equal minutes keep ascending stop order,
    // which is the order `reachable` returns them in.
    reached.sort((a, b) => a.minutes - b.minutes);

    return {
      origin: this.stopOut(source),
      departure: formatGtfsTime(departure),
      budget_minutes: minutes,
      reached,
      unreachable: result.unreachableCount(),
    };
  }

  /** The full label table for one origin, for the console's own use. */
  arrivals(query: IsochroneQuery): { departure: number; best: Int32Array } {
    const source = this.resolve(query.from, "origin");
    const departure = parseTime(query.time ?? DEFAULTS.time);
    const result = new Raptor(this.timetable, {
      minChangeTime: query.changeTime ?? DEFAULTS.changeTime,
    }).run(source, departure, { maxWalkSeconds: query.maxWalk ?? null });
    return { departure, best: result.best };
  }

  private journeyOut(journey: Journey): JourneyOut {
    return {
      departure: formatGtfsTime(journey.departure),
      arrival: formatGtfsTime(journey.arrival),
      duration_minutes: toMinutes(journeyDuration(journey)),
      transfers: journeyTransfers(journey),
      walk_minutes: toMinutes(journeyWalkSeconds(journey)),
      legs: journey.legs.map((leg) => this.legOut(leg)),
    };
  }

  private legOut(leg: Leg): LegOut {
    const stops = this.timetable.stops;
    const from = stops[leg.fromStop];
    const to = stops[leg.toStop];
    if (leg.kind === "ride") {
      return {
        kind: "ride",
        from_stop: from.stopId,
        to_stop: to.stopId,
        from_name: from.name,
        to_name: to.name,
        departure: formatGtfsTime(leg.departure),
        arrival: formatGtfsTime(leg.arrival),
        route_id: leg.routeId,
        trip_id: leg.tripId,
        seconds: leg.arrival - leg.departure,
      };
    }
    return {
      kind: "walk",
      from_stop: from.stopId,
      to_stop: to.stopId,
      from_name: from.name,
      to_name: to.name,
      // Present and null, because that is what the API sends and the parity
      // suite compares whole objects.
      departure: null,
      arrival: null,
      route_id: null,
      trip_id: null,
      seconds: leg.seconds,
    };
  }
}

/** A journey's legs as a comparable string, standing in for Python's tuple equality. */
function legKey(legs: Leg[]): string {
  return legs
    .map((leg) =>
      leg.kind === "ride"
        ? `r:${leg.routeId}:${leg.tripId}:${leg.fromStop}:${leg.toStop}:${leg.departure}:${leg.arrival}`
        : `w:${leg.fromStop}:${leg.toStop}:${leg.seconds}`,
    )
    .join("|");
}

export { NEVER };
