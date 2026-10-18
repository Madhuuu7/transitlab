"""Derive walking transfers from stop geography, when a feed declares none.

Why this file exists
--------------------
RAPTOR takes the walking graph as an input. The paper assumes it is given, and
GTFS has a place to give it -- ``transfers.txt``. Real feeds frequently do not.
The BMTC feed for Bengaluru declares **zero** transfers across 9,948 stops.

Without footpaths, a passenger can only change buses where two patterns share
the *identical* ``stop_id``. That is a much harsher constraint than it sounds,
because agencies model each direction of a road as its own stop: the BMTC feed
has 9,960 stop ids over 5,211 distinct stop names, so roughly half of all
stops are one side of a street whose other side is a separate id 40-80 m away.
Severing those pairs does not merely lose a few options -- it reroutes the
answer. Majestic to Shivajinagara came back as three bus changes because the
router was not allowed to cross a road.

So when a feed is silent, the walking graph is inferred from coordinates. This
is what OpenTripPlanner and friends do, and it is an *estimate*: straight-line
distance over an assumed walking speed, with no knowledge of whether there is
actually a footpath, a signal, or a flyover in the way. It is wrong in detail
and much closer to right than assuming nobody can walk anywhere.

Why a grid and not all pairs
----------------------------
All pairs is 9,948^2 = 99 million distance computations for a mid-size city,
and quadratic in a quantity that grows. Stops are bucketed into a grid of
cell size equal to the search radius, so a stop can only have a neighbour
within radius if that neighbour is in its own cell or one of the eight
adjacent ones. That makes the work proportional to the number of stops times
the local stop density -- linear in practice, and the constant is small
because the candidate set per stop is tens of stops rather than thousands.

Why the radius is small
-----------------------
The default is 250 m, which is deliberately tight. The job here is to reconnect
the two sides of a road and the platforms of one station, not to model a
willingness to walk across a neighbourhood. The cost of a generous radius is
paid twice: the walking graph gets denser, and RAPTOR relaxes footpaths once
per round, so every extra edge is re-walked in every round of every query.
A 600 m radius in central Bengaluru produces an order of magnitude more edges
for journeys no passenger would accept.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

from .timetable import Stop

#: Straight-line metres within which two stops are considered walkable.
DEFAULT_RADIUS_M: float = 250.0

#: Metres per second. 1.25 m/s (4.5 km/h) is the usual planning figure for an
#: unhurried adult, and is already generous for a crowded Indian footpath.
DEFAULT_WALK_SPEED_MPS: float = 1.25

#: Floor on any generated transfer. A 40 m hop across a road is not a 32-second
#: operation in practice: there is a signal, or traffic, or a median to find a
#: gap in. Without a floor the router treats cross-street transfers as nearly
#: free and produces itineraries that only work for someone who sprints.
DEFAULT_MIN_SECONDS: int = 60

#: Detour factor applied to straight-line distance. Walking follows roads, not
#: the hypotenuse; 1.4 is the common planning multiplier for a dense street
#: grid. Without it every generated time is optimistic by about a third.
DEFAULT_DETOUR: float = 1.4


def generate_footpaths(
    stops: Sequence[Stop],
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    walk_speed_mps: float = DEFAULT_WALK_SPEED_MPS,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    detour: float = DEFAULT_DETOUR,
    exclude: Iterable[tuple[str, str]] = (),
) -> list[tuple[str, str, int]]:
    """Walking transfers between stops within ``radius_m`` of each other.

    Returns ``(from_stop_id, to_stop_id, seconds)`` in both directions for
    every qualifying pair. Pairs in ``exclude`` are skipped, which is how a
    feed's own ``transfers.txt`` keeps precedence over a guess: an agency that
    has published a transfer time knows something about that interchange --
    a tunnel, a fare gate, a minimum connection time -- that coordinates
    cannot express.

    Self-pairs are not emitted; the timetable's closure adds the trivial
    self-link itself.
    """
    if radius_m <= 0:
        return []
    if walk_speed_mps <= 0:
        raise ValueError("walk_speed_mps must be positive")

    coordinates = [(s.lat, s.lon) for s in stops]
    if not coordinates:
        return []

    # Project to local metres. A city spans well under a degree, so a
    # tangent-plane approximation about the feed's mean latitude is accurate
    # to centimetres here -- far below the error in assuming straight-line
    # walking at all, and it makes the distance test a plain subtraction.
    mean_lat = sum(lat for lat, _ in coordinates) / len(coordinates)
    metres_per_deg_lat = 111_132.0
    metres_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))

    xs = [lon * metres_per_deg_lon for _, lon in coordinates]
    ys = [lat * metres_per_deg_lat for lat, _ in coordinates]

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x, y) in enumerate(zip(xs, ys)):
        buckets[(int(x // radius_m), int(y // radius_m))].append(index)

    excluded = {(a, b) for a, b in exclude}
    radius_squared = radius_m * radius_m
    out: list[tuple[str, str, int]] = []

    for (cell_x, cell_y), members in buckets.items():
        # Only half the neighbourhood is scanned. Pairs are symmetric, so
        # visiting (cell, cell + east) also settles (cell + east, cell), and
        # scanning both would emit every pair twice.
        candidates: list[int] = []
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1)):
            if dx or dy:
                candidates.extend(buckets.get((cell_x + dx, cell_y + dy), ()))

        for position, i in enumerate(members):
            xi, yi = xs[i], ys[i]
            # Within the home cell, compare against later members only, for
            # the same reason: each unordered pair exactly once.
            for j in (*members[position + 1:], *candidates):
                dx = xs[j] - xi
                dy = ys[j] - yi
                distance_squared = dx * dx + dy * dy
                if distance_squared > radius_squared:
                    continue
                seconds = max(
                    min_seconds,
                    int(round(math.sqrt(distance_squared) * detour / walk_speed_mps)),
                )
                a, b = stops[i].stop_id, stops[j].stop_id
                if (a, b) not in excluded:
                    out.append((a, b, seconds))
                if (b, a) not in excluded:
                    out.append((b, a, seconds))

    return out
