/**
 * The browser router must answer what the server answers.
 *
 * That is the one claim the deployed console rests on. It has no backend, so
 * nothing at runtime can catch it being wrong -- a mis-ported comparison
 * would not throw, it would quietly hand someone a journey that is a few
 * minutes or one change away from the truth.
 *
 * So the Python API's responses are recorded by `scripts/record_parity.py`
 * over the real Bengaluru feed -- 49 journeys and 6 isochrones, seeded, with
 * unreachable pairs, hour-of-day variation, three change times and a walking
 * limit -- and this compares whole response objects against them. Not arrival
 * times: whole objects, including every leg, every route and trip id, and the
 * nulls on a walk leg. Two routers can agree on when you arrive and disagree
 * about which bus you were on.
 *
 * The fixture is regenerated in CI from the pushed commit, so it cannot drift
 * away from either implementation unnoticed.
 */

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { decodeBundle, type BundleManifest, type BundleStrings } from "../src/engine/bundle.ts";
import { LocalEngine, type IsochroneResponse, type JourneyResponse } from "../src/engine/api.ts";
import { Timetable } from "../src/engine/timetable.ts";

const here = dirname(fileURLToPath(import.meta.url));
const FEED_DIR = join(here, "..", "public", "feed");
const FIXTURE = join(here, "fixtures", "parity.json");

interface JourneyCase {
  query: {
    from: string;
    to: string;
    time: string;
    change_time: number;
    max_walk: number | null;
  };
  response: JourneyResponse;
}

interface IsochroneCase {
  query: {
    from: string;
    time: string;
    minutes: number;
    change_time: number;
    max_walk: number | null;
  };
  record: {
    reached_count: number;
    unreachable: number;
    minutes_sum: number;
    origin: unknown;
    departure: string;
    budget_minutes: number;
    reached?: IsochroneResponse["reached"];
    first?: IsochroneResponse["reached"];
    last?: IsochroneResponse["reached"];
  };
}

interface Fixture {
  seed: number;
  feed: {
    feed: string;
    stops: number;
    patterns: number;
    trips: number;
    stop_times: number;
    footpaths: number;
  };
  journeys: JourneyCase[];
  isochrones: IsochroneCase[];
  digest_edge: number;
}

function loadBundle(): Timetable {
  const manifest = JSON.parse(
    gunzipSync(readFileSync(join(FEED_DIR, "manifest.json.gz"))).toString("utf8"),
  ) as BundleManifest;
  const strings = JSON.parse(
    gunzipSync(readFileSync(join(FEED_DIR, "strings.json.gz"))).toString("utf8"),
  ) as BundleStrings;
  const binary = gunzipSync(readFileSync(join(FEED_DIR, "timetable.bin.gz")));
  const buffer = binary.buffer.slice(
    binary.byteOffset,
    binary.byteOffset + binary.byteLength,
  ) as ArrayBuffer;
  return decodeBundle(manifest, buffer, strings);
}

const fixture = JSON.parse(readFileSync(FIXTURE, "utf8")) as Fixture;
const timetable = loadBundle();
const engine = new LocalEngine(timetable, fixture.feed.feed);

describe("the bundle carries the feed the server loaded", () => {
  it("reports the same counts as /feed", () => {
    expect(engine.feed()).toEqual(fixture.feed);
  });

  it("decodes stop coordinates exactly, not approximately", () => {
    // Floats here are the snapshot's float64 bytes, so this is an equality
    // test on purpose. A lossy coordinate would move a stop on the map and
    // nothing would complain.
    for (const journey of fixture.journeys.slice(0, 10)) {
      const origin = journey.response.origin;
      const index = timetable.stopIndex.get(origin.stop_id)!;
      expect(timetable.stops[index].lat).toBe(origin.lat);
      expect(timetable.stops[index].lon).toBe(origin.lon);
      expect(timetable.stops[index].name).toBe(origin.name);
    }
  });
});

describe("journeys match the Python API", () => {
  it("has a fixture worth trusting", () => {
    // A suite of 49 reachable pairs would not exercise the null path, and one
    // of 49 unreachable pairs would not exercise anything at all.
    const reachable = fixture.journeys.filter((c) => c.response.fastest !== null);
    const unreachable = fixture.journeys.length - reachable.length;
    expect(reachable.length).toBeGreaterThan(30);
    expect(unreachable).toBeGreaterThan(0);
    const withTransfers = reachable.filter((c) => c.response.fastest!.transfers > 0);
    const withWalks = reachable.filter((c) =>
      c.response.fastest!.legs.some((leg) => leg.kind === "walk"),
    );
    expect(withTransfers.length).toBeGreaterThan(5);
    expect(withWalks.length).toBeGreaterThan(5);
  });

  for (const [index, testCase] of fixture.journeys.entries()) {
    const { from, to, time, change_time, max_walk } = testCase.query;
    const label =
      `#${index} ${from} -> ${to} at ${time}` +
      ` (change ${change_time}s${max_walk === null ? "" : `, walk <= ${max_walk}s`})`;

    it(label, () => {
      const actual = engine.journey({
        from,
        to,
        time,
        changeTime: change_time,
        maxWalk: max_walk,
      });
      expect(actual).toEqual(testCase.response);
    });
  }
});

describe("isochrones match the Python API", () => {
  for (const [index, testCase] of fixture.isochrones.entries()) {
    const { from, time, minutes, change_time, max_walk } = testCase.query;

    it(`#${index} ${minutes} min from ${from} at ${time}`, () => {
      const actual = engine.isochrone({
        from,
        time,
        minutes,
        changeTime: change_time,
        maxWalk: max_walk,
      });

      expect(actual.origin).toEqual(testCase.record.origin);
      expect(actual.departure).toBe(testCase.record.departure);
      expect(actual.budget_minutes).toBe(testCase.record.budget_minutes);
      expect(actual.unreachable).toBe(testCase.record.unreachable);
      expect(actual.reached.length).toBe(testCase.record.reached_count);

      // The sum over every arrival is the sharp part of the digest: it cannot
      // stay equal while any one stop's arrival moves.
      const sum = actual.reached.reduce((total, stop) => total + stop.minutes, 0);
      expect(sum).toBe(testCase.record.minutes_sum);

      if (testCase.record.reached) {
        expect(actual.reached).toEqual(testCase.record.reached);
      } else {
        const edge = fixture.digest_edge;
        expect(actual.reached.slice(0, edge)).toEqual(testCase.record.first);
        expect(actual.reached.slice(-edge)).toEqual(testCase.record.last);
      }
    });
  }
});
