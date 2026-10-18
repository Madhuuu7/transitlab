from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import small_random_timetable
from transitlab.gtfs.snapshot import (
    _ARRAYS,
    SNAPSHOT_VERSION,
    SnapshotError,
    load,
    save,
)
from transitlab.gtfs.times import parse_gtfs_time
from transitlab.routing.raptor import Raptor


def round_trip(timetable, tmp_path, name="feed.npz"):
    return load(save(timetable, tmp_path / name))


# ----------------------------------------------------------------------
# the arrays survive exactly
# ----------------------------------------------------------------------
def test_every_array_round_trips_bit_for_bit(sample, tmp_path) -> None:
    restored = round_trip(sample, tmp_path)
    for name, dtype in _ARRAYS.items():
        original = getattr(sample, name)
        copy = getattr(restored, name)
        assert copy.dtype == dtype, name
        np.testing.assert_array_equal(original, copy, err_msg=name)


def test_stops_round_trip_with_ids_names_and_coordinates(sample, tmp_path) -> None:
    restored = round_trip(sample, tmp_path)
    assert restored.stops == sample.stops


def test_string_tables_round_trip(sample, tmp_path) -> None:
    restored = round_trip(sample, tmp_path)
    assert restored.pattern_route_id == sample.pattern_route_id
    assert restored.trip_ids == sample.trip_ids


def test_stop_index_is_rebuilt_and_consistent(sample, tmp_path) -> None:
    """It is derived, not stored, so it cannot drift from the stop list."""
    restored = round_trip(sample, tmp_path)
    assert restored.stop_index == sample.stop_index
    for index, stop in enumerate(restored.stops):
        assert restored.stop_index[stop.stop_id] == index


def test_derived_scalar_caches_are_rebuilt(sample, tmp_path) -> None:
    """__post_init__ must run on the restored object, not just on a built one."""
    restored = round_trip(sample, tmp_path)
    for pattern in range(sample.n_patterns):
        assert restored.pattern_length(pattern) == sample.pattern_length(pattern)
        assert restored.trip_count(pattern) == sample.trip_count(pattern)


def test_sizes_match(sample, tmp_path) -> None:
    restored = round_trip(sample, tmp_path)
    assert (restored.n_stops, restored.n_patterns, restored.n_trips, restored.n_stop_times) == (
        sample.n_stops, sample.n_patterns, sample.n_trips, sample.n_stop_times
    )


# ----------------------------------------------------------------------
# a snapshot must route identically -- the only property that matters
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_routing_is_identical_after_a_round_trip(seed, tmp_path) -> None:
    """Every arrival at every stop from every origin, before and after.

    Equal arrays are necessary but not sufficient: a snapshot that lost a
    derived index or reordered a pattern could still compare equal field by
    field and answer differently. So the check is on answers.
    """
    original = small_random_timetable(seed)
    restored = round_trip(original, tmp_path, f"feed{seed}.npz")

    before = Raptor(original, min_change_time=120)
    after = Raptor(restored, min_change_time=120)
    departure = parse_gtfs_time("08:00:00")

    for stop in original.stop_index:
        expected = before.run(stop, departure)
        actual = after.run(stop, departure)
        np.testing.assert_array_equal(
            expected.label_tables()[0],
            actual.label_tables()[0],
            err_msg=f"arrival labels differ from origin {stop}",
        )


def test_journeys_are_identical_after_a_round_trip(sample, tmp_path) -> None:
    restored = round_trip(sample, tmp_path)
    departure = parse_gtfs_time("08:00:00")
    before = Raptor(sample, min_change_time=120).run("S0", departure, "S5")
    after = Raptor(restored, min_change_time=120).run("S0", departure, "S5")
    assert str(before.journey("S5")) == str(after.journey("S5"))


# ----------------------------------------------------------------------
# refuse what cannot be trusted
# ----------------------------------------------------------------------
def test_a_future_version_is_refused_not_guessed_at(sample, tmp_path) -> None:
    """A changed array layout would otherwise load fine and route wrongly."""
    path = save(sample, tmp_path / "feed.npz")
    with np.load(path, allow_pickle=False) as archive:
        members = {name: archive[name] for name in archive.files}
    meta = json.loads(str(members["meta"]))
    meta["version"] = SNAPSHOT_VERSION + 1
    members["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **members)

    with pytest.raises(SnapshotError, match="version"):
        load(path)


def test_a_foreign_npz_is_refused(tmp_path) -> None:
    path = tmp_path / "not-a-snapshot.npz"
    np.savez_compressed(path, something=np.arange(4))
    with pytest.raises(SnapshotError, match="not a transitlab snapshot"):
        load(path)


def test_a_truncated_snapshot_is_refused(sample, tmp_path) -> None:
    path = save(sample, tmp_path / "feed.npz")
    with np.load(path, allow_pickle=False) as archive:
        members = {name: archive[name] for name in archive.files if name != "departures"}
    np.savez_compressed(path, **members)

    with pytest.raises(SnapshotError, match="missing departures"):
        load(path)


def test_inconsistent_stop_metadata_is_refused(sample, tmp_path) -> None:
    path = save(sample, tmp_path / "feed.npz")
    with np.load(path, allow_pickle=False) as archive:
        members = {name: archive[name] for name in archive.files}
    meta = json.loads(str(members["meta"]))
    meta["stop_names"] = meta["stop_names"][:-1]
    members["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **members)

    with pytest.raises(SnapshotError, match="inconsistent stop metadata"):
        load(path)
