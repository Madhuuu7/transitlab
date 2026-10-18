"""Benchmark RAPTOR against the time-expanded Dijkstra baseline.

Both answer the same question -- earliest arrival at every stop from one
origin -- so the comparison is apples to apples. What differs is the work
each does to get there, which is the point of implementing the research
router instead of running Dijkstra on a graph.

Usage::

    python scripts/benchmark.py                   # synthetic city, default size
    python scripts/benchmark.py --feed path.zip    # a real GTFS feed
    python scripts/benchmark.py --queries 200 --markdown
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transitlab.gtfs.loader import LoadStats, load_timetable  # noqa: E402
from transitlab.gtfs.synthetic import random_feed  # noqa: E402
from transitlab.gtfs.timetable import Timetable, timetable_from_feed  # noqa: E402
from transitlab.routing.raptor import Raptor  # noqa: E402
from transitlab.routing.reference import TimeExpandedGraph  # noqa: E402


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start


def build_synthetic(stops: int, patterns: int, trips: int, footpaths: int) -> Timetable:
    feed = random_feed(
        7,
        n_stops=stops,
        n_patterns=patterns,
        trips_per_pattern=trips,
        n_footpaths=footpaths,
    )
    return timetable_from_feed(feed)


def memory_bytes(timetable: Timetable) -> int:
    arrays = (
        timetable.arrivals,
        timetable.departures,
        timetable.pattern_stops,
        timetable.pattern_stop_offset,
        timetable.pattern_trip_offset,
        timetable.stop_time_offset,
        timetable.stop_pattern_offset,
        timetable.stop_pattern_pattern,
        timetable.stop_pattern_position,
        timetable.footpath_offset,
        timetable.footpath_target,
        timetable.footpath_time,
    )
    return sum(int(a.nbytes) for a in arrays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, help="GTFS zip or directory")
    parser.add_argument("--date", help="service date YYYYMMDD, for a real feed")
    parser.add_argument("--stops", type=int, default=1200)
    parser.add_argument("--patterns", type=int, default=240)
    parser.add_argument("--trips", type=int, default=40)
    parser.add_argument("--footpaths", type=int, default=400)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--change-time", type=int, default=120)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="RAPTOR only; the baseline graph is the slow part on big feeds",
    )
    parser.add_argument("--markdown", action="store_true", help="emit a table")
    args = parser.parse_args()

    if args.feed:
        stats = LoadStats()
        timetable, load_seconds = timed(
            load_timetable, args.feed, date=args.date, stats=stats
        )
        source = f"{args.feed.name} ({stats})"
    else:
        timetable, load_seconds = timed(
            build_synthetic, args.stops, args.patterns, args.trips, args.footpaths
        )
        source = (
            f"synthetic: {args.stops} stops, {args.patterns} patterns, "
            f"{args.trips} trips each"
        )

    print(f"feed      {source}")
    print(f"timetable {timetable.summary()}")
    print(f"build     {load_seconds * 1000:.0f} ms")
    print(f"arrays    {memory_bytes(timetable) / 1e6:.2f} MB")

    rng = random.Random(1)
    origins = [rng.randrange(timetable.n_stops) for _ in range(args.queries)]
    departure = 8 * 3600

    raptor = Raptor(timetable, min_change_time=args.change_time)
    raptor.run(origins[0], departure)  # warm the code paths
    raptor_times = []
    for origin in origins:
        _, seconds = timed(raptor.run, origin, departure)
        raptor_times.append(seconds * 1000)

    print()
    print(f"RAPTOR    {_describe(raptor_times)}  (one-to-all, {args.queries} queries)")

    if args.skip_reference:
        return 0

    graph, graph_seconds = timed(
        TimeExpandedGraph, timetable, change_time=args.change_time
    )
    reference_times = []
    for origin in origins:
        _, seconds = timed(graph.earliest_arrivals, origin, departure)
        reference_times.append(seconds * 1000)

    print(f"Dijkstra  {_describe(reference_times)}  (same queries)")
    print()
    print(f"graph     {graph.size}, built in {graph_seconds * 1000:.0f} ms")
    speedup = statistics.median(reference_times) / statistics.median(raptor_times)
    print(f"speedup   {speedup:.1f}x on median query time")

    if args.markdown:
        print()
        print("| | RAPTOR | time-expanded Dijkstra |")
        print("|---|---|---|")
        print(
            f"| preprocessing | {load_seconds * 1000:.0f} ms (feed -> arrays) "
            f"| {graph_seconds * 1000:.0f} ms (arrays -> graph) |"
        )
        print(
            f"| structure | {memory_bytes(timetable) / 1e6:.2f} MB of arrays "
            f"| {graph.size} |"
        )
        print(
            f"| median query | {statistics.median(raptor_times):.1f} ms "
            f"| {statistics.median(reference_times):.1f} ms |"
        )
        print(f"| speedup | **{speedup:.1f}x** | 1x |")

    return 0


def _describe(samples: list[float]) -> str:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return (
        f"median {statistics.median(ordered):6.1f} ms   "
        f"mean {statistics.fmean(ordered):6.1f} ms   p95 {p95:6.1f} ms"
    )


if __name__ == "__main__":
    raise SystemExit(main())
