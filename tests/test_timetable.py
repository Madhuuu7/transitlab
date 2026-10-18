from __future__ import annotations

import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import small_random_timetable
from transitlab.gtfs.timetable import (
    MAX_FOOTPATH_SECONDS,
    OvertakingError,
    RawPattern,
    Stop,
    build_timetable,
    close_footpaths,
)
from transitlab.routing.reference import floyd_warshall_closure


def _stops(count: int) -> list[Stop]:
    return [Stop(f"S{i}", f"Stop {i}", 12.9 + i / 100, 77.5 + i / 100) for i in range(count)]


def test_patterns_keep_their_stop_order(sample) -> None:
    blue_full = [
        p
        for p in range(sample.n_patterns)
        if sample.pattern_route_id[p] == "BLUE" and sample.pattern_length(p) == 4
    ]
    assert len(blue_full) == 1
    stops = [sample.stops[i].stop_id for i in sample.stops_of(blue_full[0])]
    assert stops == ["S0", "S1", "S2", "S3"]


def test_one_route_id_becomes_several_patterns(sample) -> None:
    """The Blue line runs a full route and a short turn."""
    blue = [p for p in range(sample.n_patterns) if sample.pattern_route_id[p] == "BLUE"]
    assert len(blue) == 2
    assert sorted(sample.pattern_length(p) for p in blue) == [3, 4]


def test_stop_time_index_arithmetic(sample) -> None:
    """``offset[p] + trip * L + position`` must address the right cell."""
    for pattern in range(sample.n_patterns):
        length = sample.pattern_length(pattern)
        base = int(sample.stop_time_offset[pattern])
        for trip in range(sample.trip_count(pattern)):
            for position in range(length):
                index = base + trip * length + position
                assert sample.arrival_at(pattern, trip, position) == sample.arrivals[index]
                assert sample.departure_at(pattern, trip, position) == sample.departures[index]


def test_departure_column_is_a_view_not_a_copy(sample) -> None:
    column = sample.departure_column(0, 1)
    assert column.base is sample.departures or column.base is not None
    assert not column.flags["OWNDATA"]
    expected = [sample.departure_at(0, t, 1) for t in range(sample.trip_count(0))]
    assert column.tolist() == expected


def test_departure_columns_are_sorted(sample) -> None:
    """The precondition for binary-searching the earliest boardable trip."""
    for pattern in range(sample.n_patterns):
        for position in range(sample.pattern_length(pattern)):
            column = sample.departure_column(pattern, position)
            assert np.all(np.diff(column) >= 0)


def test_earliest_trip_picks_the_first_boardable_one(sample) -> None:
    pattern = next(
        p
        for p in range(sample.n_patterns)
        if sample.pattern_route_id[p] == "BLUE" and sample.pattern_length(p) == 4
    )
    column = sample.departure_column(pattern, 0)
    for trip, departure in enumerate(column.tolist()):
        assert sample.earliest_trip(pattern, 0, departure) == trip
        assert sample.earliest_trip(pattern, 0, departure - 1) == trip
    assert sample.earliest_trip(pattern, 0, int(column[-1]) + 1) == -1


def test_stop_to_pattern_index_is_the_exact_inverse(sample) -> None:
    forward = {
        (p, i)
        for p in range(sample.n_patterns)
        for i in range(sample.pattern_length(p))
    }
    backward = {
        (p, i) for stop in range(sample.n_stops) for p, i in sample.patterns_at(stop)
    }
    assert forward == backward
    for stop in range(sample.n_stops):
        for pattern, position in sample.patterns_at(stop):
            assert int(sample.stops_of(pattern)[position]) == stop


def test_trips_are_sorted_by_departure_even_if_the_feed_is_not() -> None:
    raw = RawPattern(
        route_id="R",
        stops=["S0", "S1"],
        trip_ids=["late", "early"],
        arrivals=[[1000, 1200], [600, 800]],
        departures=[[1000, 1200], [600, 800]],
    )
    timetable = build_timetable(_stops(2), [raw])
    assert timetable.trip_id_of(0, 0) == "early"
    assert timetable.departure_column(0, 0).tolist() == [600, 1000]


def test_overtaking_pattern_is_rejected() -> None:
    """A later-departing trip that arrives first breaks the binary search."""
    raw = RawPattern(
        route_id="EXPRESS_AND_LOCAL",
        stops=["S0", "S1"],
        trip_ids=["local", "express"],
        arrivals=[[600, 2400], [700, 1000]],
        departures=[[600, 2400], [700, 1000]],
    )
    with pytest.raises(OvertakingError, match="overtake"):
        build_timetable(_stops(2), [raw])


def test_times_may_not_run_backwards_along_a_trip() -> None:
    raw = RawPattern(
        route_id="R",
        stops=["S0", "S1"],
        trip_ids=["t"],
        arrivals=[[600, 300]],
        departures=[[600, 300]],
    )
    with pytest.raises(ValueError, match="backwards"):
        build_timetable(_stops(2), [raw])


def test_unknown_stop_in_a_pattern_is_an_error() -> None:
    raw = RawPattern(
        route_id="R",
        stops=["S0", "GHOST"],
        trip_ids=["t"],
        arrivals=[[600, 700]],
        departures=[[600, 700]],
    )
    with pytest.raises(ValueError, match="unknown stop"):
        build_timetable(_stops(2), [raw])


def test_footpath_closure_chains_two_walks(sample) -> None:
    """S2->S4 and S4->S2 are declared; a chain must appear in the closure."""
    s2 = sample.stop_index["S2"]
    s4 = sample.stop_index["S4"]
    assert dict(sample.footpaths_from(s2)) == {s4: 300}
    assert dict(sample.footpaths_from(s4)) == {s2: 300}


def test_closure_excludes_self_links() -> None:
    offset, target, _time = close_footpaths(3, [(0, 1, 60), (1, 2, 60)], 600)
    assert 0 not in target[offset[0] : offset[1]].tolist()


def test_closure_respects_the_bound() -> None:
    # 0 -> 1 -> 2 is 700s in total, past a 600s budget, so 2 stays unreachable
    offset, target, times = close_footpaths(3, [(0, 1, 400), (1, 2, 300)], 600)
    reachable = dict(zip(target[offset[0] : offset[1]].tolist(), times[offset[0] : offset[1]].tolist()))
    assert reachable == {1: 400}


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_closure_matches_floyd_warshall(seed: int) -> None:
    """Bounded Dijkstra per source must equal the cubic all-pairs answer."""
    rng = random.Random(seed)
    n_stops = rng.randint(2, 9)
    edges = []
    for _ in range(rng.randint(1, 14)):
        u = rng.randrange(n_stops)
        v = rng.randrange(n_stops)
        if u != v:
            edges.append((u, v, rng.randrange(60, 500, 30)))

    offset, target, times = close_footpaths(n_stops, edges, MAX_FOOTPATH_SECONDS)
    mine = {
        (src, int(target[k])): int(times[k])
        for src in range(n_stops)
        for k in range(int(offset[src]), int(offset[src + 1]))
    }
    assert mine == floyd_warshall_closure(n_stops, edges, MAX_FOOTPATH_SECONDS)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=500))
def test_random_feeds_build_and_stay_consistent(seed: int) -> None:
    timetable = small_random_timetable(seed)
    assert timetable.n_stop_times == sum(
        timetable.pattern_length(p) * timetable.trip_count(p)
        for p in range(timetable.n_patterns)
    )
    assert int(timetable.stop_time_offset[-1]) == timetable.n_stop_times
    assert int(timetable.pattern_stop_offset[-1]) == timetable.pattern_stops.size
    assert timetable.footpath_offset[-1] == timetable.footpath_target.size
