# transitlab

A public-transit journey planner where the routing algorithm is the project.

**[Plan a journey across Bengaluru →](https://madhuuu7.github.io/transitlab/)**
Live, with no server behind it: the page downloads the city's timetable once
and runs the router in your tab.

Most transit apps call a directions API. This implements **RAPTOR**
(Round-bAsed Public Transit Optimized Router, Delling–Pajor–Werneck 2012) from
scratch over real GTFS data, proves it correct against an independently written
time-expanded Dijkstra, and benchmarks it against that same baseline.

```
 GTFS zip ──► DuckDB ingest ──► flat CSR arrays ──► RAPTOR ──► FastAPI
              patterns, times   offset[p]+trip*L+i   rounds     journeys
                                        │
                                        └──► 1.9 MB bundle ──► the same RAPTOR,
                                                               in the browser
```

## What it does

- **Earliest-arrival journeys** between any two stops, with walking transfers.
- **Alternatives by transfer count** — the Pareto front over (arrival time,
  transfers), read out of the round table at no extra search cost.
- **Isochrones** — "everywhere reachable in 30 minutes", which is the same
  one-to-all run looked at differently.
- **Service-day handling** — pick services by date from `calendar.txt` plus
  `calendar_dates.txt` exceptions.

## Results

RAPTOR versus the time-expanded Dijkstra baseline, both pure Python on the
same machine, same question (earliest arrival at *every* stop from one
origin), median over 15–30 random origins:

| network | stop-times | RAPTOR | Dijkstra baseline | speedup |
|---|---|---|---|---|
| 1,200 stops, 240 patterns | 47,120 | **11.9 ms** | 91.7 ms | **7.7x** |
| 4,000 stops, 700 patterns | 171,450 | **13.3 ms** | 313.2 ms | **23.5x** |

The gap widens with network size: the baseline must build and traverse a graph
with two nodes per stop-time (342,900 nodes / 666,239 edges on the larger
feed), while RAPTOR's structure stays 1.5 MB of flat arrays and its query cost
scales with rounds, not with graph size.

```sh
python scripts/benchmark.py --stops 4000 --patterns 700 --queries 15
```

## The router runs in the browser

The deployed console has no backend. It downloads the timetable once and
answers every query in the tab, using a TypeScript transcription of the same
RAPTOR implementation — same rounds, same two labels per stop, same pruning,
same journey reconstruction.

That started as a deployment problem and turned out to be the faster answer.
Both engines, same feed, same work, 40 one-to-all queries at 08:00 with a
120-second change time:

| | Python | browser |
|---|---|---|
| median | 1,373 ms | **37.1 ms** |
| mean | 1,437 ms | 37.9 ms |
| p95 | 2,394 ms | 53.8 ms |
| stops reached, mean | 9,414 / 9,948 | 9,552 / 9,948 |

Not a NumPy story. RAPTOR's round loop is irreducibly scalar — millions of
individual `int32` reads — and the arrays were already flat, so there is
nothing left to vectorise. The 37x is what a JIT does to that loop and an
interpreter does not. Both numbers come out of CI on every push.

### Which means the port has to be provable

A mis-ported comparison would not throw. It would hand someone a journey three
minutes or one change away from the truth, and look entirely plausible doing
it — and the live page has no server to be caught out by.

So `scripts/record_parity.py` drives the real FastAPI app over the Bengaluru
snapshot and records what a client would actually receive: 49 journeys and 6
isochrones, seeded, with 10 unreachable pairs, six departure times, four change
times and two walking limits. `web/test/parity.test.ts` answers the same
queries in TypeScript and compares **whole response objects** — every leg,
every route and trip id, the `null`s on a walk leg. Two routers can agree on
when you arrive and disagree about which bus you were on.

Three things had to be *made* to agree rather than assumed to:

- **Ties.** Two patterns can reach a stop at the same second, and a label only
  improves on a strict `<`, so whichever is scanned first becomes the recorded
  parent — the journey you get back depends on scan order. The Python iterated
  a `set`, leaving that to the hash table. Both sides now sort.
- **Rounding.** The API reports `round(seconds / 60)`, and Python rounds halves
  to even: a 150-second walk is 2 minutes there and 3 under `Math.round`.
- **The linear-scan threshold.** Kept at 24 trips, because the finding behind
  it is about the feed, not about Python.

## Sending a city over the wire

A snapshot is the right shape to load from disk and the wrong shape to send to
a browser. Two properties of the real feed, both measured by the encoder rather
than assumed, take 7.86 MB to **1.90 MB**:

- **The feed records no dwell time.** `arrivals` and `departures` are equal at
  all 1,540,565 stop-times, because the agency publishes one time per stop. So
  half the stop-time data is a copy of the other half. The encoder compares the
  two arrays and ships one table when they match, both when they do not.
- **Stop times fit in 16 bits.** Absolute GTFS seconds need `int32` because the
  service day runs past 24:00, but measured from its own trip's first departure
  every stop time here is at most 44,999 seconds. Storing `uint16` offsets
  beside one `int32` per trip halves the bytes and, more importantly, makes
  them compress — neighbouring offsets are close numbers, where absolute
  timestamps spread over a day are not. 3.32 MB → 0.77 MB. Above 65,535 the
  format falls back to `int32` rather than wrapping, and a test stretches a
  trip past the limit to prove it.

The decoder rebuilds the full `int32` tables before the router sees them, so
the encoding is invisible above that layer and the browser runs the same
arithmetic the server does.

## Why it is built this way

**Flat CSR arrays, not objects.** Every stop-time is an `int32` in one array,
addressed as `offset[pattern] + trip * L + position`. The inner loop of RAPTOR
walks a pattern's stops in order, so those reads are sequential; a list of
`StopTime` objects would be a pointer chase per stop.

**Patterns, not GTFS routes.** RAPTOR's "route" is a maximal set of trips over
an *identical stop sequence*. One `route_id` routinely holds several — a full
run and a short turn, or both directions — so patterns are extracted in SQL
with `string_agg(stop_id ORDER BY stop_sequence)` per trip. Grouping by
`route_id` instead breaks the algorithm's assumption that any trip of a route
can be boarded at any of its stops.

**Trip lookup by linear scan, not binary search.** This one was a mistake,
corrected by real data — see [What real data changed](#what-real-data-changed).
The earliest boardable trip is read from `departures[base+i : base+T*L : L]`,
a stride-`L` column. Searching it with `np.searchsorted` is right when a
pattern has many trips and badly wrong when it has two, which is the real
case. Below 24 trips the column is scanned in a Python loop over an `int32`
memoryview; above it, `searchsorted`. Either way the lookup depends on the
non-overtaking invariant, which the loader *verifies* and, where a feed
violates it (an express sharing a stop sequence with a local), repairs by
splitting the pattern into bands.

**Bounded footpath closure.** RAPTOR relaxes walking links once per round, so
the walking graph must be transitively closed up front. Closure is all-pairs
shortest path, but one bounded Dijkstra per source keeps each search inside a
few nearby stops — Floyd–Warshall at 10k stops is a trillion operations and a
400 MB matrix. (Floyd–Warshall *is* in the test suite, on toy graphs, as the
oracle for the closure.)

**GTFS times never wrap at 86400.** Hours above 24 are legal: a trip leaving
23:50 and arriving 00:20 is written `24:20:00` to keep it on one service day.
Reducing modulo a day turns that 30-minute trip into a 23.5-hour one.

## Correctness

The router is not trusted; it is checked against something that shares none of
its machinery. `routing/reference.py` builds an explicit time-expanded event
graph and runs Dijkstra over it — no rounds, no patterns, no binary search, no
non-overtaking assumption. Hypothesis generates random feeds, and every origin
is compared one-to-all against the reference.

That found two real bugs that reading the code did not:

1. **A single label per stop loses journeys.** Textbook RAPTOR prunes any
   arrival that does not improve a stop's label. That is sound only if a stop
   reached *on foot* may walk onward as freely as one reached *by vehicle* —
   and with a bounded closure it may not, because `A→B→C` can be missing while
   `A→B` is present. Walking to a stop at 04:06 was suppressing the bus that
   got there at 06:38, and the onward walk only the bus arrival was entitled
   to make never happened, so a stop looked unreachable when it was not. Fixed
   by labelling each stop twice (`tau_v` arrived-by-vehicle, `tau_f`
   arrived-on-foot) and relaxing footpaths only out of `tau_v`.
2. **The reference let passengers change vehicles for free.** Its
   "stay on board" edge fed into the stop's waiting chain, so a rider could
   arrive, stay seated, and step onto a later departure without paying the
   change time. Only varying the change time exposed it — the bug was
   invisible at `change_time = 0`.

Both have named regression tests. Journeys are additionally re-derived leg by
leg from the timetable, rather than trusted because the router emitted them.

```
108 tests: times, CSR index arithmetic, closure vs Floyd–Warshall, loader
quirks, RAPTOR vs reference, generated footpaths vs brute-force all-pairs,
snapshot round-trips, API contract
```

## What real data changed

Everything above was built and benchmarked against generated feeds. Then the
real [BMTC feed for Bengaluru](ATTRIBUTION.md) went in — 9,948 stops, 7,838
patterns, 1.54 million stop-times — and it is worth being specific about what
survived contact and what did not.

**It loaded with no code changes**, which is the one part that went well. The
defensive quirk handling all earned its keep: 121 trips dropped for unusable
times, string stop ids like `20921_PF5`, non-standard column order, no
`calendar_dates.txt`, and **365 patterns split because real buses genuinely
overtake each other** — an edge case that had only ever been exercised by a
test I wrote by hand.

Three things were wrong.

**1. The feed declares no walking transfers, and that changes answers.**
`transfers.txt` is absent, so the router could only change buses where two
patterns shared an identical `stop_id`. Agencies model each direction of a
road as its own stop — 9,960 ids over 5,211 distinct names here — so roughly
half of all stops are one side of a street whose other side is a separate id
40–80 m away. Forbidding that crossing does not trim the options, it reroutes
the answer:

| 08:00 departure | no footpaths | generated footpaths |
|---|---|---|
| Majestic → Shivajinagara | 27 min, 3 changes | **16 min, 1 change** |
| Majestic → Electronic City | 45 min, 2 changes | **38 min, 1 change** |
| Banashankari → Electronic City | 121 min, 5 changes | **53 min, 2 changes** |
| stops reachable in 60 min, random origin | 48 | **869** |

So `gtfs/footpaths.py` infers the walking graph from coordinates when a feed
is silent: stops bucketed into a grid of cell size equal to the search radius,
nine cells searched instead of all pairs, 0.02 s for 9,948 stops rather than
99 million distance computations. It is checked against a brute-force
all-pairs implementation, because a wrong grid offset would silently drop
boundary-straddling pairs and merely leave a feed slightly less connected.
The generated times are an estimate and say so: straight line, 1.4 detour
factor, 1.25 m/s, 60-second floor, no knowledge of whether a crossing exists.

**2. The binary search was the wrong trade, and the synthetic benchmark could
not have shown it.** The README used to advertise `np.searchsorted` over the
strided departure column. In this feed the **median pattern has 2 trips** and
74% have four or fewer — a pattern is a distinct stop sequence, and a bus
route runs a handful of variants rather than one sequence at a regular
headway. Every lookup was paying a slice object and a NumPy dispatch, several
microseconds, to binary-search two elements. The generated feeds gave every
pattern a regular headway and dozens of trips, which is precisely the shape
that hides this.

```
one-to-all over all 9,948 stops:   5.0 s  ->  1.4 s   (3.5x)
```

What remains is the interpreter in the round loop itself, which needs
vectorisation or a compiled kernel rather than a cheaper accessor. The
differential suite against the Dijkstra oracle passes unchanged, which is the
only reason a hot-path rewrite like this is defensible at all.

**3. It could not be deployed, for a reason that had nothing to do with
routing.** Ingest peaks near **1.6 GB** resident and takes 15 s, to produce
**15.4 MB** of flat arrays. A free instance has 512 MB, so the server could
not build a city it can serve with room to spare. Ingest is now a build step:
`gtfs/snapshot.py` writes the finished arrays and reads them back with no
DuckDB, no CSV parsing and no intermediate Python objects.

```
startup        15.0 s  ->  0.14 s   (107x)
steady memory   976 MB ->   96 MB   (10x)
peak memory    1588 MB ->  100 MB   (16x)
data shipped     44 MB ->  7.9 MB
```

The honest summary: the algorithm and the data structures held up, the
*benchmark* did not. A synthetic feed built to be convenient was convenient in
exactly the ways that flattered the implementation, and two of the three
problems above were invisible until a real agency's data went through it.

## Run it

```sh
python -m venv .venv && .venv/Scripts/pip install -e ".[api,dev]"

python scripts/make_sample_feed.py          # writes data/sample-feed.zip
python -m pytest                            # 138 tests
uvicorn transitlab.api.app:app --reload     # http://127.0.0.1:8000
```

### The console

It needs no API — only the timetable bundle, which is committed:

```sh
cd web && npm install
npm run dev                                 # http://127.0.0.1:5177
npm test                                    # 58 parity assertions vs the Python API
node test/bench.ts                          # the browser router, on the real feed
```

Rebuild the bundle after changing the loader, the footpath generator or the
CSR layout, for the same reason the snapshot needs rebuilding:

```sh
python scripts/make_web_bundle.py data/bmtc.npz web/public/feed
python scripts/record_parity.py             # re-record the answers to match
```

Deployment is GitHub Pages, published by `.github/workflows/pages.yml` on
every push to `main`. Nothing in the published version is recorded or
degraded — it computes each query on demand, which is why the page carries no
"this is a snapshot" caveat.

### Running the real Bengaluru network

`data/bmtc.npz` ships with the repo, so this needs no download:

```sh
TRANSITLAB_FEED=data/bmtc.npz uvicorn transitlab.api.app:app
```

9,948 stops, 1.54 M stop-times, 0.14 s to start, 96 MB resident. Try
`/journey?from=20921&to=21933&time=08:00:00` — Majestic to Electronic City —
or `/isochrone?from=20921&minutes=30`, which reaches 1,891 stops.

### Any other feed

| variable | what |
|---|---|
| `TRANSITLAB_FEED` | a GTFS zip or directory, or a `.npz` snapshot |
| `TRANSITLAB_DATE` | `20260115` — keep only services running that day |
| `TRANSITLAB_WALK_RADIUS_M` | `250` — generate footpaths; needed if the feed has no `transfers.txt` |

A city-sized feed should be ingested once and served from a snapshot, not
re-ingested on every boot:

```sh
python scripts/make_snapshot.py path/to/feed.zip data/city.npz --walk-radius 250
```

### API

| endpoint | what |
|---|---|
| `GET /journey?from=&to=&time=&change_time=` | fastest journey + alternatives by transfer count |
| `GET /isochrone?from=&time=&minutes=` | every stop reachable in the budget |
| `GET /stops?q=` | stop list, name-filtered |
| `GET /feed` | loaded feed summary |
| `GET /docs` | OpenAPI UI |

Journey queries on the full Bengaluru network take ~0.6 s; a one-to-all
isochrone over all 9,948 stops takes ~1.4 s. Both are pure Python.

### The sample feed

`gtfs/synthetic.py` emits a *real GTFS zip*, so the production loader is
exercised end to end and the tests stay offline. The seven-stop sample carries
one deliberate trap each: a route that splits into two patterns, a trip that
crosses midnight, and a pair of stops joined only by a footpath, so the fast
journey needs a transfer between two vehicle legs. `random_feed(seed)` builds
larger networks for property tests and benchmarks.

## Limitations

- Earliest-arrival and transfer-count criteria only; full multi-criteria
  RAPTOR (McRAPTOR) with walking as a third dimension is not implemented.
- `frequencies.txt` (frequency-expanded trips) is read but not expanded.
- Journey lines on the map are straight segments between stops;
  `shapes.txt` geometry is not loaded.
- Blank intermediate stop times are filled from their partner field rather
  than interpolated along the trip.
- **No bus numbers.** A leg tells you where to board and when, not which bus
  to catch. The loader takes `trips.route_id`, and in this feed those are
  internal numerics (`1000`, `1001`); the number painted on the bus is
  `route_short_name` in `routes.txt`, which is never read. Fixing it means a
  loader change and a rebuilt snapshot, and the upstream feed has moved on
  since this one was built, so it would also move every figure quoted here.
  The console shows no route plaque rather than a plausible-looking wrong one.
- Python queries cost hundreds of milliseconds to ~1.4 s on a city-sized
  network. The remaining cost is the interpreter in the round loop, which is
  exactly what the browser engine demonstrates: the same loop under a JIT is
  37x faster. Making the Python match would mean a compiled kernel, not tuned
  accessors.
- Generated footpaths are straight-line estimates. They do not know whether a
  crossing, a footway or a flyover is in the way, and they will cheerfully
  walk you across a six-lane road. A feed's own `transfers.txt` always wins.
- The shipped Bengaluru data is an **unofficial** feed whose timetables are
  known to be approximate, and it only contains routes with working live
  tracking. See [ATTRIBUTION.md](ATTRIBUTION.md). Good enough to prove the
  router works at real scale; not travel advice.

## Layout

```
src/transitlab/
  gtfs/times.py        GTFS time parsing, unwrapped past midnight
  gtfs/synthetic.py    deterministic feed generators that emit real zips
  gtfs/timetable.py    flat CSR arrays + bounded footpath closure
  gtfs/loader.py       DuckDB ingest, pattern extraction, overtaking split
  gtfs/footpaths.py    walking transfers from geography, grid-bucketed
  gtfs/snapshot.py     built timetables saved and reloaded, versioned
  gtfs/webbundle.py    the same timetable re-encoded for a browser, 7.9 -> 1.9 MB
  routing/raptor.py    the router, journey reconstruction, isochrones
  routing/reference.py time-expanded Dijkstra oracle and baseline
  api/app.py           FastAPI endpoints
web/src/engine/        the same RAPTOR in TypeScript, over typed arrays
  raptor.ts              a transcription of routing/raptor.py
  timetable.ts           the CSR accessors, same arithmetic
  bundle.ts              decodes the bundle back to full int32 tables
  api.ts                 the HTTP endpoints, answered locally
web/src/ui/            the console: map, stop picker, journey diagram
web/test/parity.test.ts  holds the browser router to the Python one
scripts/               sample feed, snapshot, bundle, parity, benchmark
```
