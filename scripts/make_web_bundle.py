"""Build the browser routing bundle from a snapshot or a GTFS feed.

    python scripts/make_web_bundle.py data/bmtc.npz web/public/feed

The console loads what this writes and runs RAPTOR over it in the tab, so
rebuild it whenever the loader, the footpath generator or the CSR layout
changes -- the same rule as the snapshot, for the same reason.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transitlab.gtfs import webbundle  # noqa: E402
from transitlab.gtfs.loader import load_timetable  # noqa: E402
from transitlab.gtfs.snapshot import load as load_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feed", type=Path, help="a .npz snapshot or a GTFS .zip")
    parser.add_argument(
        "out",
        type=Path,
        nargs="?",
        default=Path("web/public/feed"),
        help="directory to write manifest.json.gz, timetable.bin.gz, strings.json.gz",
    )
    args = parser.parse_args()

    if not args.feed.exists():
        print(f"no such feed: {args.feed}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    if args.feed.suffix == ".npz":
        timetable = load_snapshot(args.feed)
    else:
        timetable, _ = load_timetable(args.feed)
    loaded = time.perf_counter() - started

    bundle = webbundle.encode(timetable, feed_name=args.feed.name)
    sizes = webbundle.write(bundle, args.out)

    total = sum(sizes.values())
    source = args.feed.stat().st_size
    print(f"{args.feed.name}: {timetable.summary()}")
    print(f"loaded in {loaded:.2f}s")
    print(
        f"encoding {bundle.manifest['encoding']}, "
        f"dwell times {'kept' if bundle.manifest['dwell'] else 'absent from the feed'}"
    )
    for name in sorted(sizes):
        print(f"  {name + '.gz':20s} {sizes[name] / 1e6:6.2f} MB")
    print(f"  {'total':20s} {total / 1e6:6.2f} MB  (source {source / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
