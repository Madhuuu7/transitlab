/**
 * RAPTOR in the browser, ported from `transitlab.routing.raptor`.
 *
 * This is a transcription, not a reimplementation. The rounds, the two labels
 * per stop, the order of the two stages, the pruning conditions and the
 * journey reconstruction all follow the Python line for line, because the
 * claim the deployed console makes is that it computes what the server
 * computes -- and `web/test/parity.test.ts` holds it to that against recorded
 * answers from the Python router on the real Bengaluru feed.
 *
 * Read the Python module for why any of it is shaped this way. The one thing
 * worth repeating here is the part that is easy to "simplify" into a bug:
 * each stop carries **two** labels, `tauVehicle` (best arrival whose last leg
 * was a ride) and `tauFoot` (best arrival whose last leg was a walk). A
 * single label is only sound if a stop reached on foot may walk onward as
 * freely as one reached by vehicle, and the footpath closure is bounded, so
 * it may not. Differential testing against the time-expanded Dijkstra oracle
 * is what caught that; collapsing the two here would pass every unit test and
 * silently lose journeys.
 *
 * Where this departs from the Python at all, it is in representation: the
 * per-round tables are single flat `Int32Array`s indexed `k * nStops + stop`
 * rather than 2-D arrays, and the "improved" set is a mark array plus a list
 * rather than a hash set. Neither changes a result.
 */

import { NEVER } from "./times.ts";
import type { Timetable } from "./timetable.ts";

export interface TripLeg {
  kind: "ride";
  routeId: string;
  tripId: string;
  fromStop: number;
  toStop: number;
  departure: number;
  arrival: number;
}

export interface WalkLeg {
  kind: "walk";
  fromStop: number;
  toStop: number;
  seconds: number;
}

export type Leg = TripLeg | WalkLeg;

export interface Journey {
  departure: number;
  arrival: number;
  legs: Leg[];
}

export function journeyDuration(journey: Journey): number {
  return journey.arrival - journey.departure;
}

export function journeyTransfers(journey: Journey): number {
  const rides = journey.legs.filter((leg) => leg.kind === "ride").length;
  return Math.max(rides - 1, 0);
}

export function journeyWalkSeconds(journey: Journey): number {
  let total = 0;
  for (const leg of journey.legs) if (leg.kind === "walk") total += leg.seconds;
  return total;
}

export interface RaptorOptions {
  minChangeTime?: number;
  maxRounds?: number;
  maxWalkSeconds?: number | null;
  /** Target pruning. Arrivals at other stops become lower bounds when set. */
  destination?: number | null;
}

/**
 * One-to-all result of a single run: the per-round arrival table plus the
 * parent pointers needed to reconstruct a journey to *any* stop, which is
 * what makes an isochrone nearly free once a journey has been routed.
 */
export interface RaptorTables {
  timetable: Timetable;
  origin: number;
  departureTime: number;
  roundsUsed: number;
  nStops: number;
  tau: Int32Array;
  tauVehicle: Int32Array;
  tauFoot: Int32Array;
  best: Int32Array;
  parentPattern: Int32Array;
  parentTrip: Int32Array;
  parentBoard: Int32Array;
  parentAlight: Int32Array;
  parentWalkFrom: Int32Array;
  parentWalkSecs: Int32Array;
}

export class RaptorResult {
  readonly timetable: Timetable;
  readonly origin: number;
  readonly departureTime: number;
  readonly roundsUsed: number;
  readonly nStops: number;
  readonly best: Int32Array;
  private readonly tau: Int32Array;
  private readonly tauVehicle: Int32Array;
  private readonly tauFoot: Int32Array;
  private readonly parentPattern: Int32Array;
  private readonly parentTrip: Int32Array;
  private readonly parentBoard: Int32Array;
  private readonly parentAlight: Int32Array;
  private readonly parentWalkFrom: Int32Array;
  private readonly parentWalkSecs: Int32Array;

  // Fields are assigned one by one rather than declared as constructor
  // parameter properties, and the engine avoids every other TypeScript
  // feature that needs code generated for it. The whole of it then runs
  // anywhere types can simply be stripped: in the browser through vite,
  // under vitest, and under plain `node test/bench.ts` with no build step.
  constructor(tables: RaptorTables) {
    this.timetable = tables.timetable;
    this.origin = tables.origin;
    this.departureTime = tables.departureTime;
    this.roundsUsed = tables.roundsUsed;
    this.nStops = tables.nStops;
    this.best = tables.best;
    this.tau = tables.tau;
    this.tauVehicle = tables.tauVehicle;
    this.tauFoot = tables.tauFoot;
    this.parentPattern = tables.parentPattern;
    this.parentTrip = tables.parentTrip;
    this.parentBoard = tables.parentBoard;
    this.parentAlight = tables.parentAlight;
    this.parentWalkFrom = tables.parentWalkFrom;
    this.parentWalkSecs = tables.parentWalkSecs;
  }

  /** Earliest arrival at `stop`, or {@link NEVER} if unreachable. */
  arrival(stop: number): number {
    return this.best[stop];
  }

  byRound(round: number, stop: number): number {
    return this.tau[round * this.nStops + stop];
  }

  /** Stop indices reachable within a time budget -- the isochrone set. */
  reachable(withinSeconds: number): number[] {
    const limit = this.departureTime + withinSeconds;
    const out: number[] = [];
    for (let stop = 0; stop < this.nStops; stop += 1) {
      if (this.best[stop] <= limit) out.push(stop);
    }
    return out;
  }

  /**
   * The three label tables, as `label_tables()` returns them in Python.
   *
   * Exposed for the same reason: a test that only compares arrival times can
   * pass while the two labels have drifted apart, and the two-label scheme is
   * the subtle part of this router.
   */
  labelTables(): { tau: Int32Array; tauVehicle: Int32Array; tauFoot: Int32Array } {
    return { tau: this.tau, tauVehicle: this.tauVehicle, tauFoot: this.tauFoot };
  }

  unreachableCount(): number {
    let count = 0;
    for (let stop = 0; stop < this.nStops; stop += 1) {
      if (this.best[stop] >= NEVER) count += 1;
    }
    return count;
  }

  /** The earliest-arrival journey to `stop`, or null. */
  journey(stop: number): Journey | null {
    const arrival = this.best[stop];
    if (arrival >= NEVER) return null;
    let round = 0;
    for (let k = 0; k <= this.roundsUsed; k += 1) {
      if (this.tau[k * this.nStops + stop] === arrival) {
        round = k;
        break;
      }
    }
    return this.build(stop, round, arrival);
  }

  /**
   * One journey per transfer count, dropping those no faster.
   *
   * Row k already holds "fastest with at most k trips", so keeping a round
   * only when it beats every earlier round yields the Pareto front over
   * (arrival, transfers) at no extra search cost.
   */
  journeysByTransfers(stop: number): Journey[] {
    const out: Journey[] = [];
    let incumbent = NEVER;
    for (let k = 0; k <= this.roundsUsed; k += 1) {
      const arrival = this.tau[k * this.nStops + stop];
      if (arrival >= incumbent || arrival >= NEVER) continue;
      incumbent = arrival;
      const journey = this.build(stop, k, arrival);
      if (journey && journey.legs.length) out.push(journey);
    }
    return out;
  }

  private build(target: number, round: number, arrival: number): Journey | null {
    const legs = this.walkBack(target, round);
    legs.reverse();
    return { departure: this.departureTime, arrival, legs };
  }

  /**
   * The legs of a journey, backwards from `target`.
   *
   * `byFoot` tracks which of the two labels is being followed. A walk leg must
   * have come out of a vehicle arrival (or out of the origin), so the step
   * after one always follows the vehicle label; a ride drops the round index,
   * which is what bounds the loop.
   */
  private walkBack(target: number, round: number): Leg[] {
    const tt = this.timetable;
    const S = this.nStops;
    const legs: Leg[] = [];
    let stop = target;
    let k = round;
    let byFoot = this.tauFoot[k * S + stop] === this.tau[k * S + stop];
    let guard = 0;

    for (;;) {
      if (stop === this.origin && this.parentPattern[k * S + stop] < 0) return legs;
      guard += 1;
      if (guard > 2 * (this.roundsUsed + 1) * (S + 1)) {
        throw new Error("journey reconstruction failed to terminate");
      }

      if (byFoot) {
        const source = this.parentWalkFrom[k * S + stop];
        const seconds = this.parentWalkSecs[k * S + stop];
        if (source < 0) throw new Error(`no walk parent for stop ${stop} in round ${k}`);
        legs.push({ kind: "walk", fromStop: source, toStop: stop, seconds });
        stop = source;
        byFoot = false;
        continue;
      }

      const pattern = this.parentPattern[k * S + stop];
      if (pattern < 0) {
        throw new Error(
          `no trip parent for stop ${stop} in round ${k}; labels and parent ` +
            "pointers disagree",
        );
      }
      const trip = this.parentTrip[k * S + stop];
      const board = this.parentBoard[k * S + stop];
      const alight = this.parentAlight[k * S + stop];
      const source = tt.stopsOf(pattern)[board];
      const departure = tt.departureAt(pattern, trip, board);
      legs.push({
        kind: "ride",
        routeId: tt.patternRouteId[pattern],
        tripId: tt.tripIdOf(pattern, trip),
        fromStop: source,
        toStop: stop,
        departure,
        arrival: tt.arrivalAt(pattern, trip, alight),
      });
      stop = source;
      k -= 1;
      if (stop === this.origin && this.parentPattern[k * S + stop] < 0) return legs;
      // Which label did this boarding use? Either is a valid predecessor as
      // long as it was early enough; prefer the walk, since a foot label is
      // what a transfer over a footpath looks like.
      const foot = this.tauFoot[k * S + stop];
      byFoot = foot < NEVER && foot <= departure;
    }
  }
}

/** Earliest-arrival RAPTOR over a prepared {@link Timetable}. */
export class Raptor {
  readonly timetable: Timetable;
  readonly minChangeTime: number;
  readonly maxRounds: number;

  constructor(
    timetable: Timetable,
    options: { minChangeTime?: number; maxRounds?: number } = {},
  ) {
    this.timetable = timetable;
    this.minChangeTime = options.minChangeTime ?? 0;
    this.maxRounds = options.maxRounds ?? 8;
    if (this.maxRounds < 1) throw new Error("maxRounds must be at least 1");
  }

  run(
    origin: number | string,
    departureTime: number,
    options: RaptorOptions = {},
  ): RaptorResult {
    const tt = this.timetable;
    const S = tt.nStops;
    const source = tt.resolve(origin);
    const target = options.destination ?? null;
    const change = options.minChangeTime ?? this.minChangeTime;
    const rounds = options.maxRounds ?? this.maxRounds;
    const maxWalk = options.maxWalkSeconds ?? null;

    const size = (rounds + 1) * S;
    const tau = new Int32Array(size).fill(NEVER);
    const tauVehicle = new Int32Array(size).fill(NEVER);
    const tauFoot = new Int32Array(size).fill(NEVER);
    const best = new Int32Array(S).fill(NEVER);
    const bestVehicle = new Int32Array(S).fill(NEVER);
    const bestFoot = new Int32Array(S).fill(NEVER);

    const parentPattern = new Int32Array(size).fill(-1);
    const parentTrip = new Int32Array(size).fill(-1);
    const parentBoard = new Int32Array(size).fill(-1);
    const parentAlight = new Int32Array(size).fill(-1);
    const parentWalkFrom = new Int32Array(size).fill(-1);
    const parentWalkSecs = new Int32Array(size);

    // The passenger starts standing at the origin, so it counts as
    // walk-capable (tauVehicle) and needs no change time to board (tauFoot).
    // Both, in other words.
    tau[source] = departureTime;
    tauVehicle[source] = departureTime;
    tauFoot[source] = departureTime;
    best[source] = departureTime;
    bestVehicle[source] = departureTime;
    bestFoot[source] = departureTime;

    // "Improved last round" as a mark array plus a list: membership in O(1)
    // without hashing, and iteration in insertion order.
    const mark = new Uint8Array(S);
    let improved: number[] = [source];
    mark[source] = 1;

    const footLo = tt.footpathOffset;
    for (let k = footLo[source]; k < footLo[source + 1]; k += 1) {
      const neighbour = tt.footpathTarget[k];
      const seconds = tt.footpathTime[k];
      if (maxWalk !== null && seconds > maxWalk) continue;
      const candidate = departureTime + seconds;
      if (candidate < tauFoot[neighbour]) {
        tauFoot[neighbour] = candidate;
        bestFoot[neighbour] = candidate;
        if (candidate < tau[neighbour]) tau[neighbour] = candidate;
        if (candidate < best[neighbour]) best[neighbour] = candidate;
        parentWalkFrom[neighbour] = source;
        parentWalkSecs[neighbour] = seconds;
        if (!mark[neighbour]) {
          mark[neighbour] = 1;
          improved.push(neighbour);
        }
      }
    }

    let roundsUsed = 0;
    const queuePattern: number[] = [];
    const queuePosition = new Int32Array(tt.nPatterns);

    for (let k = 1; k <= rounds; k += 1) {
      const row = k * S;
      const previous = (k - 1) * S;
      // Carry round k-1 forward *with its parents*, so every label in row k
      // has a reconstructable predecessor and row k means "at most k trips".
      tau.copyWithin(row, previous, previous + S);
      tauVehicle.copyWithin(row, previous, previous + S);
      tauFoot.copyWithin(row, previous, previous + S);
      parentPattern.copyWithin(row, previous, previous + S);
      parentTrip.copyWithin(row, previous, previous + S);
      parentBoard.copyWithin(row, previous, previous + S);
      parentAlight.copyWithin(row, previous, previous + S);
      parentWalkFrom.copyWithin(row, previous, previous + S);
      parentWalkSecs.copyWithin(row, previous, previous + S);

      // Stage 1: each pattern through an improved stop, entered at the
      // earliest such position. Entering earlier cannot help: nothing before
      // it improved last round.
      queuePattern.length = 0;
      // Sorted for the reason the Python sorts: two patterns can reach a stop
      // at the same second, a label only improves on a strict <, so the
      // pattern scanned first becomes the recorded parent and the journey
      // handed back depends on the order. Both sides sort so that "the same
      // arrival" also means "the same legs".
      improved.sort((a, b) => a - b);
      for (const stop of improved) {
        const hi = tt.stopPatternOffset[stop + 1];
        for (let i = tt.stopPatternOffset[stop]; i < hi; i += 1) {
          const pattern = tt.stopPatternPattern[i];
          const position = tt.stopPatternPosition[i];
          if (queuePosition[pattern] === 0) {
            // Positions are stored as position + 1 so that zero means absent
            // and the queue needs no second array to clear.
            queuePosition[pattern] = position + 1;
            queuePattern.push(pattern);
          } else if (position + 1 < queuePosition[pattern]) {
            queuePosition[pattern] = position + 1;
          }
        }
      }
      if (!queuePattern.length) break;

      for (const stop of improved) mark[stop] = 0;
      improved = [];
      const rodeIn: number[] = [];
      let bound = target !== null ? best[target] : NEVER;

      for (const pattern of queuePattern) {
        const start = queuePosition[pattern] - 1;
        queuePosition[pattern] = 0;
        const stops = tt.stopsOf(pattern);
        const length = stops.length;
        let trip = -1;
        let board = -1;

        for (let position = start; position < length; position += 1) {
          const stop = stops[position];

          if (trip >= 0) {
            const arrival = tt.arrivalAt(pattern, trip, position);
            const ceiling = bestVehicle[stop] < bound ? bestVehicle[stop] : bound;
            if (arrival < ceiling) {
              tauVehicle[row + stop] = arrival;
              bestVehicle[stop] = arrival;
              if (arrival < tau[row + stop]) tau[row + stop] = arrival;
              if (arrival < best[stop]) best[stop] = arrival;
              parentPattern[row + stop] = pattern;
              parentTrip[row + stop] = trip;
              parentBoard[row + stop] = board;
              parentAlight[row + stop] = position;
              if (!mark[stop]) {
                mark[stop] = 1;
                improved.push(stop);
              }
              rodeIn.push(stop);
              if (target !== null && stop === target && arrival < bound) bound = arrival;
            }
          }

          // Boarding uses the *previous* round's labels: arriving here in this
          // round means arriving by this very scan, and boarding again would
          // not be a transfer. The change time is charged only after a ride --
          // a walking transfer has already paid its own time.
          let ready = tauFoot[previous + stop];
          const riding = tauVehicle[previous + stop];
          if (riding < NEVER && riding + change < ready) ready = riding + change;
          if (ready >= NEVER) continue;
          if (trip < 0 || ready <= tt.departureAt(pattern, trip, position)) {
            const candidate = tt.earliestTrip(pattern, position, ready);
            if (candidate >= 0 && (trip < 0 || candidate < trip)) {
              trip = candidate;
              board = position;
            }
          }
        }
      }

      // Stage 2: one footpath pass, out of the stops a vehicle just reached.
      // Transitive closure means a single pass cannot miss a walk-then-walk
      // connection that is within the bound; a chain that exceeds the bound is
      // deliberately not a connection.
      for (const stop of rodeIn) {
        const arrival = tauVehicle[row + stop];
        const hi = footLo[stop + 1];
        for (let i = footLo[stop]; i < hi; i += 1) {
          const neighbour = tt.footpathTarget[i];
          const seconds = tt.footpathTime[i];
          if (maxWalk !== null && seconds > maxWalk) continue;
          const candidate = arrival + seconds;
          const ceiling = bestFoot[neighbour] < bound ? bestFoot[neighbour] : bound;
          if (candidate < ceiling) {
            tauFoot[row + neighbour] = candidate;
            bestFoot[neighbour] = candidate;
            if (candidate < tau[row + neighbour]) tau[row + neighbour] = candidate;
            if (candidate < best[neighbour]) best[neighbour] = candidate;
            parentWalkFrom[row + neighbour] = stop;
            parentWalkSecs[row + neighbour] = seconds;
            if (!mark[neighbour]) {
              mark[neighbour] = 1;
              improved.push(neighbour);
            }
          }
        }
      }

      if (!improved.length) break;
      roundsUsed = k;
    }

    return new RaptorResult({
      timetable: tt,
      origin: source,
      departureTime,
      roundsUsed,
      nStops: S,
      tau,
      tauVehicle,
      tauFoot,
      best,
      parentPattern,
      parentTrip,
      parentBoard,
      parentAlight,
      parentWalkFrom,
      parentWalkSecs,
    });
  }

  /** Convenience one-to-one query. */
  journey(
    origin: number | string,
    destination: number | string,
    departureTime: number,
    options: RaptorOptions = {},
  ): Journey | null {
    const target = this.timetable.resolve(destination);
    const result = this.run(origin, departureTime, { ...options, destination: target });
    return result.journey(target);
  }
}
