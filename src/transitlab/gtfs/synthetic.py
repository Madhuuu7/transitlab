"""Deterministic synthetic GTFS feeds.

Two reasons this exists instead of a downloaded city feed:

1. Tests must run offline and in milliseconds. A real feed is tens of MB and
   turns a unit test into an integration test.
2. Property tests need *many* feeds. :func:`random_feed` builds networks from
   a fixed seed, so a failure is reproducible.

Both generators emit a real GTFS zip, so the production loader is exercised
end to end rather than being handed pre-digested records.

Non-overtaking invariant
------------------------
RAPTOR assumes trips serving the same stop sequence do not overtake one
another: if trip A leaves the first stop before trip B, it reaches every later
stop before B too. That is what lets the router binary-search for the earliest
boardable trip. Both generators satisfy it by construction -- every trip of a
pattern shares one travel-time profile and differs only by a constant headway
offset.
"""

from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .times import format_gtfs_time

SERVICE_ID = "WEEKDAY"


@dataclass
class Pattern:
    """One stop sequence plus the departure times that run over it."""

    route_id: str
    stops: list[str]
    hops: list[int]
    dwell: int
    departures: list[int]
    headsign: str = ""


@dataclass
class Feed:
    stops: dict[str, tuple[str, float, float]] = field(default_factory=dict)
    routes: dict[str, tuple[str, str]] = field(default_factory=dict)
    patterns: list[Pattern] = field(default_factory=list)
    transfers: list[tuple[str, str, int]] = field(default_factory=list)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        agency = "agency_id,agency_name,agency_url,agency_timezone\n"
        agency += "TL,TransitLab,https://example.invalid,Asia/Kolkata\n"

        stops_txt = "stop_id,stop_name,stop_lat,stop_lon\n"
        for stop_id, (name, lat, lon) in self.stops.items():
            stops_txt += f"{stop_id},{name},{lat:.6f},{lon:.6f}\n"

        routes_txt = "route_id,agency_id,route_short_name,route_long_name,route_type\n"
        for route_id, (short, long) in self.routes.items():
            routes_txt += f"{route_id},TL,{short},{long},3\n"

        trips_txt = "route_id,service_id,trip_id,trip_headsign\n"
        stop_times_txt = "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"

        for p_idx, pattern in enumerate(self.patterns):
            for t_idx, start in enumerate(pattern.departures):
                trip_id = f"{pattern.route_id}_p{p_idx}_t{t_idx}"
                trips_txt += (
                    f"{pattern.route_id},{SERVICE_ID},{trip_id},{pattern.headsign}\n"
                )

                clock = start
                last_seq = len(pattern.stops) - 1
                for seq, stop_id in enumerate(pattern.stops):
                    if seq == 0:
                        arrival = departure = clock
                    else:
                        clock += pattern.hops[seq - 1]
                        arrival = clock
                        departure = (
                            arrival if seq == last_seq else arrival + pattern.dwell
                        )
                        clock = departure
                    stop_times_txt += (
                        f"{trip_id},{format_gtfs_time(arrival)},"
                        f"{format_gtfs_time(departure)},{stop_id},{seq}\n"
                    )

        calendar_txt = (
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
            "start_date,end_date\n"
            f"{SERVICE_ID},1,1,1,1,1,0,0,20260101,20261231\n"
        )

        members = {
            "agency.txt": agency,
            "stops.txt": stops_txt,
            "routes.txt": routes_txt,
            "trips.txt": trips_txt,
            "stop_times.txt": stop_times_txt,
            "calendar.txt": calendar_txt,
        }

        if self.transfers:
            transfers_txt = (
                "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n"
            )
            for src, dst, secs in self.transfers:
                transfers_txt += f"{src},{dst},2,{secs}\n"
            members["transfers.txt"] = transfers_txt

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, body in members.items():
                zf.writestr(name, body)
        return path


def _every(start: int, end: int, headway: int) -> list[int]:
    return list(range(start, end + 1, headway))


def sample_feed() -> Feed:
    """A seven-stop network small enough to reason about by hand.

    Deliberate wrinkles, each of which breaks a naive implementation:

    * The Blue route runs two different stop sequences (a full run and a short
      turn), so one GTFS route_id must split into two RAPTOR patterns.
    * The Night Owl trip departs 23:50 and arrives at 24:05, so times past
      midnight must survive parsing without wrapping.
    * S2 and S4 are joined only by a footpath, so the fast way from S0 to S5
      requires relaxing a transfer between two vehicle legs.
    """
    feed = Feed()
    feed.stops = {
        "S0": ("Hilltop", 12.990, 77.540),
        "S1": ("Junction", 12.975, 77.560),
        "S2": ("Market", 12.965, 77.585),
        "S3": ("Riverside", 12.950, 77.610),
        "S4": ("Stadium", 12.962, 77.600),
        "S5": ("Airport", 13.020, 77.690),
        "S6": ("Museum", 12.940, 77.555),
    }
    feed.routes = {
        "BLUE": ("1", "Blue Line"),
        "GREEN": ("2", "Green Line"),
        "RED": ("3", "Red Line"),
        "OWL": ("N1", "Night Owl"),
    }

    h = 3600
    feed.patterns = [
        Pattern("BLUE", ["S0", "S1", "S2", "S3"], [300, 360, 420], 30,
                _every(6 * h, 10 * h, 600), "Riverside"),
        Pattern("BLUE", ["S0", "S1", "S2"], [300, 360], 30,
                _every(6 * h + 300, 9 * h, 1800), "Market"),
        Pattern("GREEN", ["S3", "S4", "S5"], [480, 720], 30,
                _every(6 * h + 300, 10 * h, 900), "Airport"),
        Pattern("RED", ["S1", "S6", "S5"], [1200, 1500], 30,
                _every(6 * h + 120, 10 * h, 1800), "Airport"),
        Pattern("OWL", ["S0", "S1", "S2"], [300, 360], 30,
                [23 * h + 3000], "Market"),
    ]
    feed.transfers = [
        ("S2", "S4", 300),
        ("S4", "S2", 300),
    ]
    return feed


def random_feed(
    seed: int,
    n_stops: int = 40,
    n_patterns: int = 12,
    trips_per_pattern: int = 20,
    n_footpaths: int = 15,
) -> Feed:
    """A random but non-overtaking network, for property tests and benchmarks."""
    rng = random.Random(seed)
    feed = Feed()

    for i in range(n_stops):
        feed.stops[f"S{i}"] = (f"Stop {i}", 12.9 + i * 0.002, 77.5 + i * 0.003)

    for p in range(n_patterns):
        length = rng.randint(2, min(8, n_stops))
        stops = rng.sample(sorted(feed.stops), length)
        hops = [rng.randrange(120, 900, 30) for _ in range(length - 1)]
        headway = rng.choice([300, 600, 900, 1200])
        first = rng.randrange(5 * 3600, 8 * 3600, 300)
        route_id = f"R{p}"
        feed.routes[route_id] = (str(p), f"Route {p}")
        feed.patterns.append(
            Pattern(
                route_id=route_id,
                stops=stops,
                hops=hops,
                dwell=rng.choice([0, 30, 60]),
                departures=[first + k * headway for k in range(trips_per_pattern)],
            )
        )

    seen: set[tuple[str, str]] = set()
    for _ in range(n_footpaths):
        a, b = rng.sample(sorted(feed.stops), 2)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        secs = rng.randrange(60, 600, 30)
        feed.transfers.append((a, b, secs))
        feed.transfers.append((b, a, secs))

    return feed


def write_sample_feed(path: str | Path) -> Path:
    return sample_feed().write(Path(path))


def write_random_feed(path: str | Path, seed: int, **kwargs) -> Path:
    return random_feed(seed, **kwargs).write(Path(path))
