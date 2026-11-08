/**
 * How long a query takes in the browser engine, on the real feed.
 *
 *     node test/bench.ts
 *
 * Not a test -- a number for the README that can be re-measured rather than
 * remembered. Node 22.18+ and 24 strip the types, so there is no build step
 * and no extra dependency to run it.
 *
 * It measures the same work `scripts/benchmark.py` measures on the Python
 * side: a one-to-all run from a given origin, which is what both a journey
 * and an isochrone cost, since one run answers every destination.
 */

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { decodeBundle, type BundleManifest, type BundleStrings } from "../src/engine/bundle.ts";
import { Raptor } from "../src/engine/raptor.ts";
import { parseGtfsTime } from "../src/engine/times.ts";

const here = dirname(fileURLToPath(import.meta.url));
const FEED_DIR = join(here, "..", "public", "feed");

const RUNS = 40;
const WARMUP = 5;

function percentile(sorted: number[], fraction: number): number {
  const index = Math.min(sorted.length - 1, Math.floor(fraction * sorted.length));
  return sorted[index];
}

const decodeStarted = performance.now();
const manifest = JSON.parse(
  gunzipSync(readFileSync(join(FEED_DIR, "manifest.json.gz"))).toString("utf8"),
) as BundleManifest;
const strings = JSON.parse(
  gunzipSync(readFileSync(join(FEED_DIR, "strings.json.gz"))).toString("utf8"),
) as BundleStrings;
const compressed = readFileSync(join(FEED_DIR, "timetable.bin.gz"));
const binary = gunzipSync(compressed);
const buffer = binary.buffer.slice(
  binary.byteOffset,
  binary.byteOffset + binary.byteLength,
) as ArrayBuffer;
const timetable = decodeBundle(manifest, buffer, strings);
const decodeMs = performance.now() - decodeStarted;

const transferred =
  compressed.byteLength +
  readFileSync(join(FEED_DIR, "strings.json.gz")).byteLength +
  readFileSync(join(FEED_DIR, "manifest.json.gz")).byteLength;

console.log(`feed      ${manifest.feed}: ${timetable.summary()}`);
console.log(`download  ${(transferred / 1e6).toFixed(2)} MB gzipped`);
console.log(`decode    ${decodeMs.toFixed(0)} ms to a routable timetable`);

const router = new Raptor(timetable, { minChangeTime: 120 });
const departure = parseGtfsTime("08:00:00");

// Spread the origins over the feed instead of hammering one well-connected
// stop, which would flatter the numbers.
const origins = Array.from(
  { length: RUNS + WARMUP },
  (_, i) => Math.floor((i * timetable.nStops) / (RUNS + WARMUP)),
);

for (let i = 0; i < WARMUP; i += 1) router.run(origins[i], departure);

const samples: number[] = [];
let reachedTotal = 0;
for (let i = WARMUP; i < origins.length; i += 1) {
  const started = performance.now();
  const result = router.run(origins[i], departure);
  samples.push(performance.now() - started);
  reachedTotal += timetable.nStops - result.unreachableCount();
}
samples.sort((a, b) => a - b);

const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
console.log(
  `one-to-all  ${RUNS} origins: mean ${mean.toFixed(1)} ms, ` +
    `median ${percentile(samples, 0.5).toFixed(1)} ms, ` +
    `p95 ${percentile(samples, 0.95).toFixed(1)} ms, ` +
    `max ${samples[samples.length - 1].toFixed(1)} ms`,
);
console.log(
  `            ${Math.round(reachedTotal / RUNS)} stops reached on average ` +
    `of ${timetable.nStops}`,
);
