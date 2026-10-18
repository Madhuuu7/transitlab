# transitlab

A public-transit journey planner where the routing algorithm is the project.

Most transit apps call a directions API. This implements **RAPTOR**
(Round-bAsed Public Transit Optimized Router, Delling–Pajor–Werneck 2012) from
scratch over real GTFS data, proves it correct against an independently written
time-expanded Dijkstra, and benchmarks it against that same baseline.

```
 GTFS zip ──► DuckDB ingest ──► flat CSR NumPy arrays ──► RAPTOR ──► FastAPI ──► MapLibre UI
              patterns, times     offset[p]+trip*L+i      rounds      journeys     map + isochrones
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

**Binary search over a strided view.** The earliest boardable trip comes from
`np.searchsorted` over `departures[base+i : base+T*L : L]` — a stride-`L`
view, so no allocation and no Python loop over trips. Valid only under the
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
75 tests: times, CSR index arithmetic, closure vs Floyd–Warshall, loader
quirks, RAPTOR vs reference, API contract
```

## Run it

```sh
python -m venv .venv && .venv/Scripts/pip install -e ".[api,dev]"

python scripts/make_sample_feed.py          # writes data/sample-feed.zip
python -m pytest                            # 75 tests
uvicorn transitlab.api.app:app --reload     # http://127.0.0.1:8000
```

Point it at a real feed with `TRANSITLAB_FEED=path/to/feed.zip` and
`TRANSITLAB_DATE=20260115`.

### API

| endpoint | what |
|---|---|
| `GET /journey?from=&to=&time=&change_time=` | fastest journey + alternatives by transfer count |
| `GET /isochrone?from=&time=&minutes=` | every stop reachable in the budget |
| `GET /stops?q=` | stop list, name-filtered |
| `GET /feed` | loaded feed summary |
| `GET /docs` | OpenAPI UI |

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

## Layout

```
src/transitlab/
  gtfs/times.py        GTFS time parsing, unwrapped past midnight
  gtfs/synthetic.py    deterministic feed generators that emit real zips
  gtfs/timetable.py    flat CSR arrays + bounded footpath closure
  gtfs/loader.py       DuckDB ingest, pattern extraction, overtaking split
  routing/raptor.py    the router, journey reconstruction, isochrones
  routing/reference.py time-expanded Dijkstra oracle and baseline
  api/app.py           FastAPI endpoints
web/index.html         MapLibre UI, no build step
scripts/               sample feed, benchmark
```
