"""Record the HTTP API's answers so the browser router can be held to them.

    python scripts/record_parity.py

The console routes in the tab. Nothing about that is interesting unless it
computes what the server computes, and "I ported it carefully" is not a
claim anyone should accept -- including me. So this walks the real Bengaluru
feed with a seeded set of queries, asks the *running FastAPI app* through its
test client, and writes the responses to ``web/test/fixtures/parity.json``.
``web/test/parity.test.ts`` then loads the browser bundle, answers the same
queries with the TypeScript engine and compares whole responses.

Recording through the app rather than calling the router directly is the
point: what gets compared is the JSON a client would receive, so a difference
in rounding, in field names or in how a walk leg is serialised shows up here
rather than in a bug report about journeys that look slightly wrong.

Isochrones over a city reach thousands of stops, which is too much to commit
in full for every query. Small budgets are stored whole; larger ones are
stored as a count, an unreachable count, the sum of every arrival in minutes
and the first and last fifty entries. The sum is what makes that sharp: it
cannot stay equal while any single arrival moves.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEED = ROOT / "data" / "bmtc.npz"
OUT = ROOT / "web" / "test" / "fixtures" / "parity.json"

#: Fixed, so the fixture is a property of the feed and not of the day it ran.
SEED = 20261108

#: Journeys sampled uniformly from the feed's stops. Most random pairs in a
#: city are reachable, and the unreachable ones are worth having too: a
#: `null` fastest journey is a response the console has to render.
N_RANDOM_JOURNEYS = 44

#: Journeys named in the README, so the numbers quoted there are themselves
#: under test on both sides.
#:
#: The spellings are the feed's, not the ones a Bengalurean would type: BMTC
#: transliterates, so it is "White Field", "Shivajinagara" and
#: "Yeshwanthapura". Looking up by the common English name finds nothing,
#: which is worth knowing about any feed before building a search box over it.
NAMED_JOURNEYS = [
    ("Majestic", "Electronic City", "08:00:00", 120, None),
    ("Banashankari", "Electronic City", "08:00:00", 120, None),
    ("Majestic", "White Field Bus Station", "18:30:00", 120, None),
    ("Shivajinagara Bus Station", "Yeshwanthapura", "07:15:00", 0, None),
    ("Jayanagara", "Hebbala", "09:45:00", 300, 600),
]

FULL_ISOCHRONE_MINUTES = 10
DIGEST_EDGE = 50


def find_stop(stops, needle: str) -> str | None:
    """The first stop whose name contains ``needle``, case-insensitively."""
    lowered = needle.lower()
    for stop in stops:
        if lowered in stop["name"].lower():
            return stop["stop_id"]
    return None


def isochrone_record(payload: dict, full: bool) -> dict:
    reached = payload["reached"]
    record = {
        "reached_count": len(reached),
        "unreachable": payload["unreachable"],
        "minutes_sum": sum(stop["minutes"] for stop in reached),
        "origin": payload["origin"],
        "departure": payload["departure"],
        "budget_minutes": payload["budget_minutes"],
    }
    if full:
        record["reached"] = reached
    else:
        record["first"] = reached[:DIGEST_EDGE]
        record["last"] = reached[-DIGEST_EDGE:]
    return record


def main() -> int:
    if not FEED.exists():
        print(f"no feed at {FEED}", file=sys.stderr)
        return 1

    os.environ["TRANSITLAB_FEED"] = str(FEED)
    from fastapi.testclient import TestClient  # noqa: E402

    from transitlab.api.app import create_app  # noqa: E402

    started = time.perf_counter()
    client = TestClient(create_app())

    feed = client.get("/feed")
    feed.raise_for_status()
    stops = client.get("/stops", params={"limit": 50_000}).json()
    print(f"{feed.json()['feed']}: {len(stops)} stops")

    rng = random.Random(SEED)
    queries: list[tuple[str, str, str, int, int | None]] = []

    for name_a, name_b, when, change, walk in NAMED_JOURNEYS:
        a, b = find_stop(stops, name_a), find_stop(stops, name_b)
        if a is None or b is None:
            print(f"  skipping {name_a} -> {name_b}: not in this feed")
            continue
        queries.append((a, b, when, change, walk))

    times = ["06:30:00", "08:00:00", "08:00", "12:00:00", "17:45:00", "22:10:00"]
    for _ in range(N_RANDOM_JOURNEYS):
        a = rng.choice(stops)["stop_id"]
        b = rng.choice(stops)["stop_id"]
        queries.append(
            (
                a,
                b,
                rng.choice(times),
                rng.choice([0, 60, 120, 300]),
                rng.choice([None, None, None, 300, 900]),
            )
        )

    journeys = []
    for index, (a, b, when, change, walk) in enumerate(queries, 1):
        params: dict[str, object] = {"from": a, "to": b, "time": when, "change_time": change}
        if walk is not None:
            params["max_walk"] = walk
        response = client.get("/journey", params=params)
        response.raise_for_status()
        journeys.append(
            {
                "query": {
                    "from": a,
                    "to": b,
                    "time": when,
                    "change_time": change,
                    "max_walk": walk,
                },
                "response": response.json(),
            }
        )
        if index % 10 == 0:
            print(f"  {index}/{len(queries)} journeys")

    isochrone_queries = [
        (queries[0][0], "08:00:00", FULL_ISOCHRONE_MINUTES, 120, None),
        (queries[1][0], "08:00:00", FULL_ISOCHRONE_MINUTES, 120, None),
        (queries[0][0], "08:00:00", 30, 120, None),
        (queries[0][0], "17:30:00", 45, 120, None),
        (queries[1][0], "08:00:00", 60, 0, None),
        (rng.choice(stops)["stop_id"], "09:00:00", 30, 120, 600),
    ]
    isochrones = []
    for origin, when, minutes, change, walk in isochrone_queries:
        params = {
            "from": origin,
            "time": when,
            "minutes": minutes,
            "change_time": change,
        }
        if walk is not None:
            params["max_walk"] = walk
        response = client.get("/isochrone", params=params)
        response.raise_for_status()
        isochrones.append(
            {
                "query": {
                    "from": origin,
                    "time": when,
                    "minutes": minutes,
                    "change_time": change,
                    "max_walk": walk,
                },
                "record": isochrone_record(
                    response.json(), full=minutes <= FULL_ISOCHRONE_MINUTES
                ),
            }
        )
        print(f"  isochrone {minutes} min from {origin}")

    payload = {
        "seed": SEED,
        "feed": feed.json(),
        "journeys": journeys,
        "isochrones": isochrones,
        "digest_edge": DIGEST_EDGE,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    elapsed = time.perf_counter() - started
    reachable = sum(1 for j in journeys if j["response"]["fastest"])
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(journeys)} journeys "
        f"({reachable} reachable), {len(isochrones)} isochrones, "
        f"{OUT.stat().st_size / 1e6:.2f} MB, in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
