"""The browser bundle must mean exactly the timetable it was built from.

The console routes over this bundle with no server to check it against, so a
decoding mistake would not raise -- it would quietly produce journeys that are
wrong in a way only a careful reader would notice. These tests pin the two
things that could go wrong silently: the arrays must survive the encoding
bit for bit, and the router must answer identically on the decoded copy.

The TypeScript decoder is checked separately, against recorded answers from
this Python one; see ``web/test/parity.test.ts``.
"""

from __future__ import annotations

import dataclasses
import gzip
import json

import numpy as np
import pytest

from conftest import dense_footpath_timetable, small_random_timetable
from transitlab.gtfs.timetable import Timetable
from transitlab.gtfs.webbundle import (
    BUNDLE_VERSION,
    U16_MAX,
    Bundle,
    BundleError,
    decode,
    encode,
    read,
    write,
)
from transitlab.routing.raptor import Raptor

_STORED_ARRAYS = (
    "coords",
    "pattern_stop_offset",
    "pattern_stops",
    "pattern_trip_offset",
    "stop_time_offset",
    "arrivals",
    "departures",
    "stop_pattern_offset",
    "stop_pattern_pattern",
    "stop_pattern_position",
    "footpath_offset",
    "footpath_target",
    "footpath_time",
)


def round_trip(timetable: Timetable) -> Timetable:
    return decode(encode(timetable, "test"))


def without_dwell(timetable: Timetable) -> Timetable:
    """A copy whose arrivals equal its departures, as real feeds often are."""
    return dataclasses.replace(timetable, arrivals=timetable.departures.copy())


# ----------------------------------------------------------------------
# the arrays survive
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_array_round_trips_bit_for_bit(seed: int) -> None:
    original = small_random_timetable(seed)
    restored = round_trip(original)
    for name in _STORED_ARRAYS:
        np.testing.assert_array_equal(
            getattr(original, name), getattr(restored, name), err_msg=name
        )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_stop_times_round_trip_when_the_feed_has_no_dwell(seed: int) -> None:
    """The branch that ships one time table instead of two."""
    original = without_dwell(small_random_timetable(seed))
    bundle = encode(original, "test")
    assert bundle.manifest["dwell"] is False
    restored = decode(bundle)
    np.testing.assert_array_equal(restored.arrivals, original.arrivals)
    np.testing.assert_array_equal(restored.departures, original.departures)


def test_dwell_times_are_kept_when_the_feed_has_them() -> None:
    original = small_random_timetable(7)
    assert not np.array_equal(original.arrivals, original.departures), (
        "fixture is meant to have dwell times; it no longer does"
    )
    bundle = encode(original, "test")
    assert bundle.manifest["dwell"] is True
    np.testing.assert_array_equal(decode(bundle).arrivals, original.arrivals)


def test_stops_strings_and_index_round_trip() -> None:
    original = small_random_timetable(4)
    restored = round_trip(original)
    assert restored.stops == original.stops
    assert restored.pattern_route_id == original.pattern_route_id
    assert restored.trip_ids == original.trip_ids
    assert restored.stop_index == original.stop_index


def test_derived_scalar_caches_are_rebuilt() -> None:
    original = small_random_timetable(5)
    restored = round_trip(original)
    for pattern in range(original.n_patterns):
        assert restored.pattern_length(pattern) == original.pattern_length(pattern)
        assert restored.trip_count(pattern) == original.trip_count(pattern)
        assert int(restored.stop_time_offset[pattern]) == int(
            original.stop_time_offset[pattern]
        )


# ----------------------------------------------------------------------
# the router cannot tell the difference
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_routing_over_the_decoded_bundle_is_identical(seed: int) -> None:
    original = small_random_timetable(seed)
    restored = round_trip(original)
    for origin in range(original.n_stops):
        a = Raptor(original, min_change_time=120).run(origin, 7 * 3600)
        b = Raptor(restored, min_change_time=120).run(origin, 7 * 3600)
        np.testing.assert_array_equal(a.best, b.best, err_msg=f"origin {origin}")
        np.testing.assert_array_equal(a.by_round, b.by_round)


def test_journeys_over_the_decoded_bundle_have_the_same_legs() -> None:
    """Equal arrival times are not enough: the legs must be the same legs."""
    original = dense_footpath_timetable(11)
    restored = round_trip(original)
    for origin in range(original.n_stops):
        a = Raptor(original, min_change_time=60).run(origin, 6 * 3600)
        b = Raptor(restored, min_change_time=60).run(origin, 6 * 3600)
        for target in range(original.n_stops):
            left, right = a.journey(target), b.journey(target)
            assert (left is None) == (right is None), (origin, target)
            if left is not None and right is not None:
                assert left.legs == right.legs, (origin, target)


# ----------------------------------------------------------------------
# the encoding choices, which are decisions and not assumptions
# ----------------------------------------------------------------------
def test_small_feeds_use_the_compact_time_encoding() -> None:
    manifest = encode(small_random_timetable(1), "test").manifest
    assert manifest["encoding"] == "u16-delta"
    names = {spec["name"] for spec in manifest["arrays"]}
    assert "departure_offset" in names
    assert "trip_first_departure" in names
    assert "departures" not in names


def test_a_trip_longer_than_sixteen_bits_falls_back_to_absolute_times() -> None:
    """A 19-hour trip cannot be stored as 16-bit offsets, so it must not be.

    Wrapping silently here would turn a long journey into a short one, which
    is the single worst thing this format could do.
    """
    original = small_random_timetable(2)
    departures = original.departures.copy()
    arrivals = original.arrivals.copy()
    base = int(original.stop_time_offset[0])
    last = base + original.pattern_length(0) - 1
    departures[last] = int(departures[base]) + U16_MAX + 1
    arrivals[last] = departures[last]
    stretched = dataclasses.replace(
        original, departures=departures, arrivals=arrivals
    )

    bundle = encode(stretched, "test")
    assert bundle.manifest["encoding"] == "i32-absolute"
    np.testing.assert_array_equal(decode(bundle).departures, departures)


def test_every_array_starts_on_an_eight_byte_boundary() -> None:
    """Float64Array over a buffer refuses a misaligned offset."""
    for spec in encode(small_random_timetable(3), "test").manifest["arrays"]:
        assert spec["offset"] % 8 == 0, spec["name"]


def test_the_manifest_describes_the_binary_it_ships_with() -> None:
    bundle = encode(small_random_timetable(6), "test")
    end = 0
    for spec in bundle.manifest["arrays"]:
        assert spec["offset"] >= end
        end = spec["offset"] + spec["bytes"]
    assert end <= len(bundle.binary)
    assert len(bundle.binary) - end < 8, "more than one alignment pad of slack"


def test_counts_in_the_manifest_match_the_timetable() -> None:
    original = small_random_timetable(8)
    counts = encode(original, "bmtc.npz").manifest["counts"]
    assert counts == {
        "stops": original.n_stops,
        "patterns": original.n_patterns,
        "trips": original.n_trips,
        "stop_times": original.n_stop_times,
        "footpaths": int(original.footpath_target.size),
    }


# ----------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------
def test_a_future_bundle_version_is_refused_rather_than_guessed_at() -> None:
    bundle = encode(small_random_timetable(1), "test")
    future = Bundle(
        manifest={**bundle.manifest, "version": BUNDLE_VERSION + 1},
        binary=bundle.binary,
        strings=bundle.strings,
    )
    with pytest.raises(BundleError, match="version"):
        decode(future)


def test_an_unknown_time_encoding_is_refused() -> None:
    bundle = encode(small_random_timetable(1), "test")
    broken = Bundle(
        manifest={**bundle.manifest, "encoding": "varint"},
        binary=bundle.binary,
        strings=bundle.strings,
    )
    with pytest.raises(BundleError, match="encoding"):
        decode(broken)


def test_a_truncated_binary_is_refused_rather_than_read_short() -> None:
    bundle = encode(small_random_timetable(1), "test")
    truncated = Bundle(
        manifest=bundle.manifest,
        binary=bundle.binary[: len(bundle.binary) // 2],
        strings=bundle.strings,
    )
    with pytest.raises(BundleError, match="past the end"):
        decode(truncated)


def test_a_missing_directory_is_refused() -> None:
    with pytest.raises(BundleError, match="not a bundle"):
        read("does-not-exist")


# ----------------------------------------------------------------------
# files on disk
# ----------------------------------------------------------------------
def test_written_files_round_trip_through_read(tmp_path) -> None:
    original = small_random_timetable(9)
    sizes = write(encode(original, "test"), tmp_path)
    assert set(sizes) == {"manifest.json", "timetable.bin", "strings.json"}
    assert all(size > 0 for size in sizes.values())
    restored = decode(read(tmp_path))
    np.testing.assert_array_equal(restored.departures, original.departures)
    assert restored.stops == original.stops


def test_the_three_files_are_gzipped_and_named_as_such(tmp_path) -> None:
    write(encode(small_random_timetable(1), "test"), tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["manifest.json.gz", "strings.json.gz", "timetable.bin.gz"]
    manifest = json.loads(gzip.decompress((tmp_path / "manifest.json.gz").read_bytes()))
    assert manifest["version"] == BUNDLE_VERSION


def test_rebuilding_the_same_timetable_writes_the_same_bytes(tmp_path) -> None:
    """Otherwise every rebuild is a diff and the repository cannot be read."""
    original = small_random_timetable(10)
    first, second = tmp_path / "a", tmp_path / "b"
    write(encode(original, "test"), first)
    write(encode(original, "test"), second)
    for name in ("manifest.json.gz", "timetable.bin.gz", "strings.json.gz"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_the_manifest_records_what_the_console_will_download(tmp_path) -> None:
    """The page states its own download size, so the build must know it.

    Measuring it in the browser is what does not work: a host that sets
    Content-Encoding on a .gz file has the browser inflate it first, and the
    number on the page then describes the host rather than the feed.
    """
    sizes = write(encode(small_random_timetable(12), "test"), tmp_path)
    manifest = json.loads(gzip.decompress((tmp_path / "manifest.json.gz").read_bytes()))
    assert manifest["download"] == {
        "timetable.bin": sizes["timetable.bin"],
        "strings.json": sizes["strings.json"],
    }
    for name, size in manifest["download"].items():
        assert (tmp_path / f"{name}.gz").stat().st_size == size
