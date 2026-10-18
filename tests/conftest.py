from __future__ import annotations

import pytest

from transitlab.gtfs.synthetic import random_feed, sample_feed
from transitlab.gtfs.timetable import Timetable, timetable_from_feed
from transitlab.routing.raptor import Journey, TripLeg, WalkLeg


@pytest.fixture(scope="session")
def sample() -> Timetable:
    return timetable_from_feed(sample_feed())


@pytest.fixture(scope="session")
def sample_zip(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("feeds") / "sample-feed.zip"
    return str(sample_feed().write(path))


def small_random_timetable(seed: int) -> Timetable:
    """A network small enough that the reference graph stays cheap."""
    return timetable_from_feed(
        random_feed(
            seed,
            n_stops=18,
            n_patterns=7,
            trips_per_pattern=8,
            n_footpaths=6,
        )
    )


def dense_footpath_timetable(seed: int) -> Timetable:
    """Walk-heavy network: the case where the bounded closure matters."""
    return timetable_from_feed(
        random_feed(
            seed,
            n_stops=16,
            n_patterns=6,
            trips_per_pattern=6,
            n_footpaths=16,
        )
    )


def assert_journey_consistent(
    timetable: Timetable, journey: Journey, *, min_change_time: int = 0
) -> None:
    """Re-derive a journey from the timetable instead of trusting the router.

    Checks that every leg exists in the feed with the stated times, that legs
    join end to end, that the clock never runs backwards, and that a transfer
    between two rides respects the change time.
    """
    assert journey.legs, "a journey must have at least one leg"
    clock = journey.departure
    previous_stop: str | None = None
    previous_was_ride = False

    for leg in journey.legs:
        if previous_stop is not None:
            assert leg.from_stop == previous_stop, "legs do not join up"

        if isinstance(leg, WalkLeg):
            source = timetable.stop_index[leg.from_stop]
            target = timetable.stop_index[leg.to_stop]
            walks = dict(timetable.footpaths_from(source))
            assert target in walks, f"no footpath {leg.from_stop}->{leg.to_stop}"
            assert walks[target] == leg.seconds, "footpath time does not match"
            clock += leg.seconds
            previous_was_ride = False
        elif isinstance(leg, TripLeg):
            needed = clock + (min_change_time if previous_was_ride else 0)
            assert leg.departure >= needed, (
                f"boards {leg.trip_id} at {leg.departure} but is only free "
                f"at {needed}"
            )
            _assert_ride_exists(timetable, leg)
            clock = leg.arrival
            previous_was_ride = True
        else:  # pragma: no cover - guards against a new leg type
            raise AssertionError(f"unknown leg type {type(leg)!r}")

        previous_stop = leg.to_stop

    assert clock <= journey.arrival, "journey arrives before its last leg ends"
    assert journey.arrival == clock, "reported arrival does not match the legs"


def _assert_ride_exists(timetable: Timetable, leg: TripLeg) -> None:
    source = timetable.stop_index[leg.from_stop]
    target = timetable.stop_index[leg.to_stop]
    for pattern in range(timetable.n_patterns):
        if timetable.pattern_route_id[pattern] != leg.route_id:
            continue
        for trip in range(timetable.trip_count(pattern)):
            if timetable.trip_id_of(pattern, trip) != leg.trip_id:
                continue
            stops = list(timetable.stops_of(pattern))
            for board, stop in enumerate(stops):
                if stop != source:
                    continue
                if timetable.departure_at(pattern, trip, board) != leg.departure:
                    continue
                for alight in range(board + 1, len(stops)):
                    if (
                        stops[alight] == target
                        and timetable.arrival_at(pattern, trip, alight) == leg.arrival
                    ):
                        return
    raise AssertionError(f"no such ride in the feed: {leg}")
