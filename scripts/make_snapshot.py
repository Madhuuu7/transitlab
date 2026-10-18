"""Build a timetable snapshot from a GTFS feed.

Run this where there is memory to spare -- a laptop, or a deploy build step --
and ship the result. The server then loads a city it could not have ingested.

    python scripts/make_snapshot.py data/bmtc.zip data/bmtc.npz --walk-radius 250
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transitlab.gtfs.loader import LoadStats, load_timetable  # noqa: E402
from transitlab.gtfs.snapshot import load as load_snapshot  # noqa: E402
from transitlab.gtfs.snapshot import save  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feed", help="GTFS zip or extracted directory")
    parser.add_argument("out", help="destination .npz")
    parser.add_argument(
        "--walk-radius",
        type=float,
        default=0.0,
        metavar="M",
        help="generate walking transfers between stops within M metres "
        "(needed for feeds with no transfers.txt; try 250)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYYMMDD",
        help="keep only services running on this date (default: every trip)",
    )
    args = parser.parse_args()

    stats = LoadStats()
    started = time.perf_counter()
    timetable = load_timetable(
        args.feed,
        date=args.date,
        walk_radius_m=args.walk_radius or None,
        stats=stats,
    )
    ingest = time.perf_counter() - started
    print(f"ingested {args.feed} in {ingest:.1f}s")
    print(f"  {stats}")

    started = time.perf_counter()
    written = save(timetable, args.out)
    wrote = time.perf_counter() - started
    size_mb = written.stat().st_size / 1e6
    print(f"wrote {written} ({size_mb:.1f} MB) in {wrote:.1f}s")

    # Read it straight back. A snapshot that cannot be loaded is worse than no
    # snapshot, and this is the only moment where catching that is free.
    started = time.perf_counter()
    reloaded = load_snapshot(written)
    reload_time = time.perf_counter() - started
    print(f"reloaded in {reload_time:.2f}s")

    if (reloaded.n_stops, reloaded.n_patterns, reloaded.n_stop_times) != (
        timetable.n_stops,
        timetable.n_patterns,
        timetable.n_stop_times,
    ):
        print("MISMATCH between built and reloaded timetable", file=sys.stderr)
        return 1

    print(
        f"  {reloaded.n_stops} stops, {reloaded.n_patterns} patterns, "
        f"{reloaded.n_stop_times} stop-times, "
        f"{reloaded.footpath_target.size} footpaths"
    )
    print(f"ingest {ingest:.1f}s -> load {reload_time:.2f}s "
          f"({ingest / max(reload_time, 1e-6):.0f}x faster startup)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
