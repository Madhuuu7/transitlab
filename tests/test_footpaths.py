from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from transitlab.gtfs.footpaths import (
    DEFAULT_DETOUR,
    DEFAULT_MIN_SECONDS,
    DEFAULT_WALK_SPEED_MPS,
    generate_footpaths,
)
from transitlab.gtfs.timetable import Stop

#: Metres per degree, matching the projection in the module under test.
M_PER_DEG_LAT = 111_132.0


def at(stop_id: str, *, north_m: float = 0.0, east_m: float = 0.0) -> Stop:
    """A stop offset from a fixed origin by a distance in metres."""
    lat0, lon0 = 12.97, 77.59
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    return Stop(stop_id, stop_id, lat0 + north_m / M_PER_DEG_LAT, lon0 + east_m / m_per_deg_lon)


def brute_force(
    stops: list[Stop], radius_m: float
) -> set[tuple[str, str]]:
    """Every within-radius ordered pair, by checking all of them.

    The grid in the module under test exists only to avoid this quadratic
    scan. Keeping the quadratic version here means the optimisation is checked
    against the definition it is meant to implement, rather than against
    itself -- the same reason the router is checked against a Dijkstra oracle.
    """
    mean_lat = sum(s.lat for s in stops) / len(stops)
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))
    pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(stops):
        for b in stops[i + 1:]:
            dx = (b.lon - a.lon) * m_per_deg_lon
            dy = (b.lat - a.lat) * M_PER_DEG_LAT
            if dx * dx + dy * dy <= radius_m * radius_m:
                pairs.add((a.stop_id, b.stop_id))
                pairs.add((b.stop_id, a.stop_id))
    return pairs


# ----------------------------------------------------------------------
# the basics
# ----------------------------------------------------------------------
def test_no_stops_means_no_footpaths() -> None:
    assert generate_footpaths([]) == []


def test_radius_zero_disables_generation() -> None:
    stops = [at("A"), at("B", east_m=10)]
    assert generate_footpaths(stops, radius_m=0) == []


def test_nearby_pair_is_linked_in_both_directions() -> None:
    stops = [at("A"), at("B", east_m=60)]
    result = generate_footpaths(stops, radius_m=250)
    assert {(a, b) for a, b, _ in result} == {("A", "B"), ("B", "A")}


def test_walking_time_is_symmetric() -> None:
    stops = [at("A"), at("B", east_m=200), at("C", north_m=120)]
    times = {(a, b): s for a, b, s in generate_footpaths(stops, radius_m=400)}
    for (a, b), seconds in times.items():
        assert times[(b, a)] == seconds


def test_far_pair_is_not_linked() -> None:
    stops = [at("A"), at("B", east_m=1_000)]
    assert generate_footpaths(stops, radius_m=250) == []


def test_no_self_pairs() -> None:
    stops = [at("A"), at("B", east_m=30), at("C", east_m=60)]
    assert all(a != b for a, b, _ in generate_footpaths(stops, radius_m=250))


def test_pair_is_emitted_exactly_once_per_direction() -> None:
    """The grid scans half the neighbourhood so pairs are not duplicated."""
    stops = [at(f"S{i}", east_m=40 * i) for i in range(12)]
    result = generate_footpaths(stops, radius_m=250)
    assert len(result) == len({(a, b) for a, b, _ in result})


# ----------------------------------------------------------------------
# the numbers
# ----------------------------------------------------------------------
def test_time_reflects_distance_detour_and_speed() -> None:
    stops = [at("A"), at("B", east_m=300)]
    (_, _, seconds), *_ = generate_footpaths(stops, radius_m=400)
    expected = 300 * DEFAULT_DETOUR / DEFAULT_WALK_SPEED_MPS
    assert seconds == pytest.approx(expected, abs=2)


def test_short_hops_are_floored_not_free() -> None:
    """A 20 m hop across a road is not a 22-second operation in practice."""
    stops = [at("A"), at("B", east_m=20)]
    (_, _, seconds), *_ = generate_footpaths(stops, radius_m=250)
    assert seconds == DEFAULT_MIN_SECONDS


def test_longer_walks_exceed_the_floor() -> None:
    stops = [at("A"), at("B", east_m=240)]
    (_, _, seconds), *_ = generate_footpaths(stops, radius_m=250)
    assert seconds > DEFAULT_MIN_SECONDS


def test_bad_walk_speed_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_footpaths([at("A")], walk_speed_mps=0)


# ----------------------------------------------------------------------
# declared transfers win
# ----------------------------------------------------------------------
def test_excluded_pairs_are_not_generated() -> None:
    """A feed's own transfers.txt keeps precedence over a guess."""
    stops = [at("A"), at("B", east_m=50)]
    result = generate_footpaths(stops, radius_m=250, exclude=[("A", "B")])
    assert {(a, b) for a, b, _ in result} == {("B", "A")}


# ----------------------------------------------------------------------
# the grid must agree with the definition it optimises
# ----------------------------------------------------------------------
@pytest.mark.parametrize("radius", [50.0, 120.0, 250.0, 500.0])
def test_grid_matches_brute_force_on_a_cluster(radius: float) -> None:
    """Stops on a lattice, so many pairs land exactly on cell boundaries.

    A half-neighbourhood scan that got its offsets wrong would silently drop
    the pairs that straddle a boundary, and a feed would simply come out
    slightly less connected -- the kind of bug that never raises.
    """
    stops = [
        at(f"S{i}-{j}", east_m=100 * i, north_m=100 * j)
        for i in range(7)
        for j in range(7)
    ]
    generated = {(a, b) for a, b, _ in generate_footpaths(stops, radius_m=radius)}
    assert generated == brute_force(stops, radius)


@settings(max_examples=40, deadline=None)
@given(
    offsets=st.lists(
        st.tuples(
            st.floats(min_value=-600, max_value=600),
            st.floats(min_value=-600, max_value=600),
        ),
        min_size=2,
        max_size=30,
    ),
    radius=st.floats(min_value=30, max_value=400),
)
def test_grid_matches_brute_force_on_random_layouts(offsets, radius) -> None:
    stops = [
        at(f"S{i}", east_m=east, north_m=north)
        for i, (east, north) in enumerate(offsets)
    ]
    generated = {(a, b) for a, b, _ in generate_footpaths(stops, radius_m=radius)}
    assert generated == brute_force(stops, radius)
