from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import (
    assert_journey_consistent,
    dense_footpath_timetable,
    small_random_timetable,
)
from transitlab.gtfs.times import NEVER, parse_gtfs_time
from transitlab.routing.raptor import Raptor, TripLeg, WalkLeg
from transitlab.routing.reference import TimeExpandedGraph

# RAPTOR caps the number of vehicle legs; the reference does not. Routing on
# the small test networks never needs this many, so the bound is not what the
# comparisons are measuring.
ROUNDS = 12


def test_direct_ride(sample) -> None:
    raptor = Raptor(sample)
    journey = raptor.journey("S0", "S3", parse_gtfs_time("06:10:00"))
    assert journey is not None
    assert [type(leg) for leg in journey.legs] == [TripLeg]
    assert journey.legs[0].route_id == "BLUE"
    assert journey.arrival == parse_gtfs_time("06:29:00")
    assert journey.n_transfers == 0
    assert_journey_consistent(sample, journey)


def test_transfer_over_a_footpath(sample) -> None:
    """S0 -> S5 is only fast if the walk from S2 to S4 is used."""
    raptor = Raptor(sample)
    journey = raptor.journey("S0", "S5", parse_gtfs_time("06:10:00"))
    assert journey is not None
    assert [type(leg) for leg in journey.legs] == [TripLeg, WalkLeg, TripLeg]
    assert journey.legs[1] == WalkLeg(from_stop="S2", to_stop="S4", seconds=300)
    assert journey.arrival == parse_gtfs_time("06:40:30")
    assert journey.n_transfers == 1
    assert journey.walk_seconds == 300
    assert_journey_consistent(sample, journey)


def test_waits_for_the_next_departure(sample) -> None:
    """Asking before the first trip of the day still returns that trip."""
    raptor = Raptor(sample)
    journey = raptor.journey("S0", "S1", parse_gtfs_time("04:00:00"))
    assert journey is not None
    assert journey.legs[0].departure == parse_gtfs_time("06:00:00")


def test_service_over_for_the_day(sample) -> None:
    raptor = Raptor(sample)
    result = raptor.run("S0", parse_gtfs_time("23:00:00"))
    assert result.arrival("S5") == NEVER
    assert result.journey("S5") is None


def test_post_midnight_trip_is_not_wrapped(sample) -> None:
    """The Night Owl leaves 23:50 and arrives 24:01:30 the same service day."""
    raptor = Raptor(sample)
    journey = raptor.journey("S0", "S2", parse_gtfs_time("23:50:00"))
    assert journey is not None
    assert journey.arrival == parse_gtfs_time("24:01:30")
    assert journey.duration == 11 * 60 + 30
    assert_journey_consistent(sample, journey)


def test_origin_is_reached_at_the_departure_time(sample) -> None:
    raptor = Raptor(sample)
    departure = parse_gtfs_time("06:10:00")
    result = raptor.run("S0", departure)
    assert result.arrival("S0") == departure
    assert result.journey("S0") is None or result.journey("S0").legs == ()


def test_change_time_pushes_the_transfer_later(sample) -> None:
    """A long required change can only make the journey later, never earlier."""
    departure = parse_gtfs_time("06:10:00")
    quick = Raptor(sample, min_change_time=0).run("S0", departure)
    slow = Raptor(sample, min_change_time=20 * 60).run("S0", departure)
    for stop in range(sample.n_stops):
        assert slow.best[stop] >= quick.best[stop]

    journey = Raptor(sample, min_change_time=20 * 60).journey("S0", "S5", departure)
    assert journey is not None
    assert_journey_consistent(sample, journey, min_change_time=20 * 60)


def test_isochrone_grows_with_the_budget(sample) -> None:
    result = Raptor(sample).run("S0", parse_gtfs_time("06:10:00"))
    previous: set[int] = set()
    for minutes in (5, 15, 30, 60):
        reached = set(result.reachable(minutes * 60).tolist())
        assert previous <= reached
        previous = reached
    assert sample.stop_index["S5"] in previous


def test_journeys_by_transfers_is_a_pareto_front(sample) -> None:
    result = Raptor(sample).run("S0", parse_gtfs_time("06:10:00"))
    journeys = result.journeys_by_transfers("S5")
    assert journeys
    arrivals = [j.arrival for j in journeys]
    transfers = [j.n_transfers for j in journeys]
    assert arrivals == sorted(arrivals, reverse=False) or len(arrivals) == 1
    # strictly improving arrival as transfers are allowed to grow
    assert all(b < a for a, b in zip(arrivals, arrivals[1:]))
    assert all(b > a for a, b in zip(transfers, transfers[1:]))
    for journey in journeys:
        assert_journey_consistent(sample, journey)


def test_target_pruning_does_not_change_the_answer(sample) -> None:
    departure = parse_gtfs_time("06:10:00")
    raptor = Raptor(sample)
    full = raptor.run("S0", departure)
    for stop in sample.stop_index:
        pruned = raptor.run("S0", departure, stop)
        assert pruned.arrival(stop) == full.arrival(stop)


def test_rounds_bound_the_number_of_rides(sample) -> None:
    departure = parse_gtfs_time("06:10:00")
    one_ride = Raptor(sample, max_rounds=1).run("S0", departure)
    assert one_ride.arrival("S3") == parse_gtfs_time("06:29:00")  # direct BLUE
    assert one_ride.arrival("S5") == NEVER  # needs a transfer
    assert Raptor(sample, max_rounds=2).run("S0", departure).arrival("S5") < NEVER


# ----------------------------------------------------------------------
# agreement with the independent reference
# ----------------------------------------------------------------------
def _compare(timetable, departure: int, change_time: int = 0) -> None:
    raptor = Raptor(timetable, min_change_time=change_time, max_rounds=ROUNDS)
    graph = TimeExpandedGraph(timetable, change_time=change_time)
    for origin in range(timetable.n_stops):
        mine = raptor.run(origin, departure).best.astype(np.int64)
        theirs = graph.earliest_arrivals(origin, departure)
        mismatch = np.flatnonzero(mine != theirs)
        assert mismatch.size == 0, (
            f"origin {timetable.stops[origin].stop_id} at {departure}: "
            f"stops {[timetable.stops[int(s)].stop_id for s in mismatch]} "
            f"raptor={mine[mismatch].tolist()} reference={theirs[mismatch].tolist()}"
        )


@settings(max_examples=20, deadline=None)
@given(
    departure=st.integers(min_value=4 * 3600, max_value=25 * 3600),
    change_time=st.sampled_from([0, 60, 300]),
)
def test_matches_reference_on_the_sample_feed(sample, departure: int, change_time: int) -> None:
    _compare(sample, departure, change_time)


@settings(max_examples=16, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=200),
    departure=st.integers(min_value=4 * 3600, max_value=12 * 3600),
    change_time=st.sampled_from([0, 180, 600]),
)
def test_matches_reference_on_random_feeds(
    seed: int, departure: int, change_time: int
) -> None:
    _compare(small_random_timetable(seed), departure, change_time)


@settings(max_examples=10, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=60),
    departure=st.integers(min_value=4 * 3600, max_value=10 * 3600),
    change_time=st.sampled_from([0, 180]),
)
def test_matches_reference_with_dense_footpaths(
    seed: int, departure: int, change_time: int
) -> None:
    """Many footpaths is the case where the walk bound bites.

    With a bounded closure a stop can be reached on foot long before any
    vehicle gets there, and only the vehicle arrival may walk onward. Sparse
    networks rarely produce that; dense ones do constantly.
    """
    _compare(dense_footpath_timetable(seed), departure, change_time)


def test_walk_label_does_not_suppress_a_later_vehicle_arrival() -> None:
    """Regression: a foot label used to hide the only onward walk.

    Walking from the origin reaches S11 at 04:06, far earlier than any bus.
    S6 is reachable *only* on foot from S11, and the chain origin->S11->S6
    exceeds the walking bound, so the journey has to be: ride to S11, then
    walk. A single label per stop prunes that bus arrival and loses S6
    entirely.
    """
    timetable = small_random_timetable(20)
    result = Raptor(timetable, max_rounds=ROUNDS).run("S10", 4 * 3600)
    assert result.arrival("S6") == 24_450
    journey = result.journey("S6")
    assert journey is not None
    assert isinstance(journey.legs[-1], WalkLeg)
    assert journey.legs[-1].from_stop == "S11"
    assert_journey_consistent(timetable, journey)


def test_riding_through_a_stop_pays_no_change_time(sample) -> None:
    """A change time is charged between vehicles, not for staying seated.

    BLUE runs S0 -> S1 -> S2 -> S3 without a transfer, so a 20-minute change
    requirement must not delay it at all.
    """
    departure = parse_gtfs_time("06:10:00")
    patient = Raptor(sample, min_change_time=20 * 60)
    journey = patient.journey("S0", "S3", departure)
    assert journey is not None
    assert len(journey.legs) == 1
    assert journey.arrival == parse_gtfs_time("06:29:00")


@settings(max_examples=8, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=200),
    departure=st.integers(min_value=5 * 3600, max_value=10 * 3600),
)
def test_random_journeys_are_internally_consistent(seed: int, departure: int) -> None:
    timetable = small_random_timetable(seed)
    raptor = Raptor(timetable, max_rounds=ROUNDS)
    result = raptor.run(0, departure)
    for stop in range(timetable.n_stops):
        journey = result.journey(stop)
        if journey is None or not journey.legs:
            continue
        assert journey.arrival == result.arrival(stop)
        assert_journey_consistent(timetable, journey)
