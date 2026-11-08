/**
 * The flat timetable, ported from `transitlab.gtfs.timetable`.
 *
 * Same CSR arrays, same accessors, same arithmetic -- the router above this
 * file is a transcription of the Python one, and keeping this layer identical
 * is what makes that possible. Nothing here knows about the bundle encoding;
 * `bundle.ts` has already rebuilt the full int32 tables by the time a
 * `Timetable` exists.
 */

export interface Stop {
  stopId: string;
  name: string;
  lat: number;
  lon: number;
}

/**
 * Trip count at or below which {@link Timetable.earliestTrip} scans linearly.
 *
 * Kept at the Python value, and for the same reason it exists there: in the
 * BMTC feed the median pattern has two trips, so a binary search pays its
 * setup to look at two elements. The threshold is part of the measurement
 * this project is about, not an implementation detail to drop in the port.
 */
export const LINEAR_SCAN_TRIPS = 24;

export class Timetable {
  readonly stops: Stop[];
  readonly stopIndex: Map<string, number>;
  readonly coords: Float64Array;

  readonly patternRouteId: string[];
  readonly patternStopOffset: Int32Array;
  readonly patternStops: Int32Array;
  readonly patternTripOffset: Int32Array;
  readonly tripIds: string[];
  readonly stopTimeOffset: Int32Array;
  readonly arrivals: Int32Array;
  readonly departures: Int32Array;

  readonly stopPatternOffset: Int32Array;
  readonly stopPatternPattern: Int32Array;
  readonly stopPatternPosition: Int32Array;

  readonly footpathOffset: Int32Array;
  readonly footpathTarget: Int32Array;
  readonly footpathTime: Int32Array;

  /** Derived per-pattern scalars, hoisted out of the inner loop. */
  private readonly lengths: Int32Array;
  private readonly tripCounts: Int32Array;

  constructor(fields: {
    stops: Stop[];
    coords: Float64Array;
    patternRouteId: string[];
    patternStopOffset: Int32Array;
    patternStops: Int32Array;
    patternTripOffset: Int32Array;
    tripIds: string[];
    stopTimeOffset: Int32Array;
    arrivals: Int32Array;
    departures: Int32Array;
    stopPatternOffset: Int32Array;
    stopPatternPattern: Int32Array;
    stopPatternPosition: Int32Array;
    footpathOffset: Int32Array;
    footpathTarget: Int32Array;
    footpathTime: Int32Array;
  }) {
    this.stops = fields.stops;
    this.coords = fields.coords;
    this.patternRouteId = fields.patternRouteId;
    this.patternStopOffset = fields.patternStopOffset;
    this.patternStops = fields.patternStops;
    this.patternTripOffset = fields.patternTripOffset;
    this.tripIds = fields.tripIds;
    this.stopTimeOffset = fields.stopTimeOffset;
    this.arrivals = fields.arrivals;
    this.departures = fields.departures;
    this.stopPatternOffset = fields.stopPatternOffset;
    this.stopPatternPattern = fields.stopPatternPattern;
    this.stopPatternPosition = fields.stopPatternPosition;
    this.footpathOffset = fields.footpathOffset;
    this.footpathTarget = fields.footpathTarget;
    this.footpathTime = fields.footpathTime;
    this.stopIndex = new Map(fields.stops.map((stop, index) => [stop.stopId, index]));

    const nPatterns = fields.patternStopOffset.length - 1;
    this.lengths = new Int32Array(nPatterns);
    this.tripCounts = new Int32Array(nPatterns);
    for (let p = 0; p < nPatterns; p += 1) {
      this.lengths[p] = fields.patternStopOffset[p + 1] - fields.patternStopOffset[p];
      this.tripCounts[p] = fields.patternTripOffset[p + 1] - fields.patternTripOffset[p];
    }
  }

  // ------------------------------------------------------------------
  // sizes
  // ------------------------------------------------------------------
  get nStops(): number {
    return this.stops.length;
  }

  get nPatterns(): number {
    return this.patternRouteId.length;
  }

  get nTrips(): number {
    return this.tripIds.length;
  }

  get nStopTimes(): number {
    return this.departures.length;
  }

  get nFootpaths(): number {
    return this.footpathTarget.length;
  }

  patternLength(pattern: number): number {
    return this.lengths[pattern];
  }

  tripCount(pattern: number): number {
    return this.tripCounts[pattern];
  }

  // ------------------------------------------------------------------
  // views (no copies)
  // ------------------------------------------------------------------
  stopsOf(pattern: number): Int32Array {
    return this.patternStops.subarray(
      this.patternStopOffset[pattern],
      this.patternStopOffset[pattern + 1],
    );
  }

  /**
   * Index of the first trip of `pattern` departing `position` at or after
   * `time`; -1 if the service day is over for that pattern.
   *
   * Both branches rely on the non-overtaking invariant the loader enforces:
   * departures down one position ascend in trip order, so the first hit is
   * the earliest.
   */
  earliestTrip(pattern: number, position: number, time: number): number {
    const length = this.lengths[pattern];
    const trips = this.tripCounts[pattern];
    const base = this.stopTimeOffset[pattern] + position;
    const departures = this.departures;

    if (trips <= LINEAR_SCAN_TRIPS) {
      let index = base;
      for (let trip = 0; trip < trips; trip += 1) {
        if (departures[index] >= time) return trip;
        index += length;
      }
      return -1;
    }

    let lo = 0;
    let hi = trips;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (departures[base + mid * length] < time) lo = mid + 1;
      else hi = mid;
    }
    return lo < trips ? lo : -1;
  }

  arrivalAt(pattern: number, trip: number, position: number): number {
    return this.arrivals[
      this.stopTimeOffset[pattern] + trip * this.lengths[pattern] + position
    ];
  }

  departureAt(pattern: number, trip: number, position: number): number {
    return this.departures[
      this.stopTimeOffset[pattern] + trip * this.lengths[pattern] + position
    ];
  }

  tripIdOf(pattern: number, trip: number): string {
    return this.tripIds[this.patternTripOffset[pattern] + trip];
  }

  /**
   * `(pattern, position)` for every pattern serving `stop`.
   *
   * The router does not call this: it walks `stopPatternOffset` directly,
   * because allocating a few thousand two-element arrays per round is the
   * kind of cost that does not show up in an algorithm and does show up in a
   * profile. This is for callers outside the rounds loop.
   */
  patternsAt(stop: number): Array<[number, number]> {
    const out: Array<[number, number]> = [];
    const hi = this.stopPatternOffset[stop + 1];
    for (let k = this.stopPatternOffset[stop]; k < hi; k += 1) {
      out.push([this.stopPatternPattern[k], this.stopPatternPosition[k]]);
    }
    return out;
  }

  /** `(target, walkSeconds)` for every footpath out of `stop`, excluding itself. */
  footpathsFrom(stop: number): Array<[number, number]> {
    const out: Array<[number, number]> = [];
    const hi = this.footpathOffset[stop + 1];
    for (let k = this.footpathOffset[stop]; k < hi; k += 1) {
      out.push([this.footpathTarget[k], this.footpathTime[k]]);
    }
    return out;
  }

  resolve(stop: number | string): number {
    if (typeof stop === "number") return stop;
    const index = this.stopIndex.get(stop);
    if (index === undefined) throw new Error(`unknown stop: ${stop}`);
    return index;
  }

  latOf(stop: number): number {
    return this.coords[stop * 2];
  }

  lonOf(stop: number): number {
    return this.coords[stop * 2 + 1];
  }

  summary(): string {
    return (
      `${this.nStops} stops, ${this.nPatterns} patterns, ${this.nTrips} trips, ` +
      `${this.nStopTimes} stop-times, ${this.nFootpaths} footpaths`
    );
  }
}
