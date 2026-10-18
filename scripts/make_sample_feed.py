"""Write the bundled sample GTFS feed to data/sample-feed.zip."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transitlab.gtfs.synthetic import write_sample_feed  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample-feed.zip")
    path = write_sample_feed(out)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
