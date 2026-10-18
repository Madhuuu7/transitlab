from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from transitlab.gtfs.loader import FeedError, LoadStats, load_timetable, split_overtaking
from transitlab.gtfs.synthetic import random_feed, sample_feed
from transitlab.gtfs.timetable import RawPattern, Timetable, timetable_from_feed


def _signature(timetable: Timetable) -> set[tuple]:
    """Patterns as comparable tuples, independent of pattern order."""
    out = set()
    for pattern in range(timetable.n_patterns):
        stops = tuple(timetable.stops[i].stop_id for i in timetable.stops_of(pattern))
        trips = tuple(
            tuple(
                timetable.departure_at(pattern, trip, position)
                for position in range(timetable.pattern_length(pattern))
            )
            for trip in range(timetable.trip_count(pattern))
        )
        out.add((timetable.pattern_route_id[pattern], stops, trips))
    return out


def test_zip_round_trip_matches_the_in_memory_build(sample_zip: str, sample) -> None:
    """The production path and the test shortcut must agree exactly."""
    loaded = load_timetable(sample_zip)
    assert [s.stop_id for s in loaded.stops] == [s.stop_id for s in sample.stops]
    assert _signature(loaded) == _signature(sample)
    assert loaded.n_stop_times == sample.n_stop_times


def test_random_feed_round_trip(tmp_path: Path) -> None:
    feed = random_feed(11, n_stops=20, n_patterns=6, trips_per_pattern=5, n_footpaths=5)
    path = feed.write(tmp_path / "random.zip")
    assert _signature(load_timetable(path)) == _signature(timetable_from_feed(feed))


def test_stats_report_what_happened(sample_zip: str) -> None:
    stats = LoadStats()
    load_timetable(sample_zip, stats=stats)
    assert stats.stops == 7
    assert stats.trips_read == 56
    assert stats.trips_dropped == 0
    assert stats.patterns == 5
    assert stats.footpaths_declared == 2
    assert "patterns" in str(stats)


def test_loads_from_an_extracted_directory(sample_zip: str, tmp_path: Path) -> None:
    out = tmp_path / "feed"
    out.mkdir()
    with zipfile.ZipFile(sample_zip) as zf:
        zf.extractall(out)
    assert load_timetable(out).n_patterns == 5


def test_service_date_selects_weekday_service(sample_zip: str) -> None:
    # calendar.txt in the sample feed runs Monday-Friday only
    assert load_timetable(sample_zip, date="20260115").n_patterns == 5  # Thursday
    with pytest.raises(FeedError, match="no services selected"):
        load_timetable(sample_zip, date="20260117")  # Saturday


def test_explicit_service_ids(sample_zip: str) -> None:
    assert load_timetable(sample_zip, service_ids=["WEEKDAY"]).n_patterns == 5
    with pytest.raises(FeedError, match="no trips matched"):
        load_timetable(sample_zip, service_ids=["SUNDAY"])


def test_bad_date_is_rejected(sample_zip: str) -> None:
    with pytest.raises(ValueError, match="YYYYMMDD"):
        load_timetable(sample_zip, date="2026-01-15")


def test_not_a_feed(tmp_path: Path) -> None:
    broken = tmp_path / "not-a-feed.zip"
    broken.write_bytes(b"this is not a zip")
    with pytest.raises(FeedError, match="not a zip"):
        load_timetable(broken)

    missing = tmp_path / "empty.zip"
    with zipfile.ZipFile(missing, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(FeedError, match="GTFS feed"):
        load_timetable(missing)

    with pytest.raises(FeedError, match="no such feed"):
        load_timetable(tmp_path / "absent.zip")


def _write_feed(directory: Path, **overrides: str) -> Path:
    """Minimal hand-written feed, so quirks can be injected one at a time."""
    files = {
        "stops.txt": (
            "stop_id,stop_name,stop_lat,stop_lon,location_type\n"
            "A,Alpha,12.9,77.5,0\n"
            "B,Beta,12.91,77.51,0\n"
            "C,Gamma,12.92,77.52,0\n"
        ),
        "routes.txt": "route_id,route_short_name,route_type\nR1,1,3\n",
        "trips.txt": "route_id,service_id,trip_id\nR1,WD,t1\n",
        "stop_times.txt": (
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "t1,08:00:00,08:00:00,A,1\n"
            "t1,08:10:00,08:11:00,B,2\n"
            "t1,08:20:00,08:20:00,C,3\n"
        ),
    }
    files.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (directory / name).write_text(body, encoding="utf-8")
    return directory


def test_station_rows_are_not_boardable(tmp_path: Path) -> None:
    feed = _write_feed(
        tmp_path / "stations",
        **{
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon,location_type\n"
                "A,Alpha,12.9,77.5,0\n"
                "B,Beta,12.91,77.51,0\n"
                "C,Gamma,12.92,77.52,0\n"
                "ST,Big Station,12.93,77.53,1\n"
            )
        },
    )
    timetable = load_timetable(feed)
    assert "ST" not in timetable.stop_index


def test_stop_sequence_need_not_be_consecutive(tmp_path: Path) -> None:
    """GTFS only requires stop_sequence to increase, so 1/5/9 is legal."""
    feed = _write_feed(
        tmp_path / "gaps",
        **{
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,08:00:00,08:00:00,A,1\n"
                "t1,08:10:00,08:11:00,B,5\n"
                "t1,08:20:00,08:20:00,C,9\n"
            )
        },
    )
    timetable = load_timetable(feed)
    assert [timetable.stops[i].stop_id for i in timetable.stops_of(0)] == ["A", "B", "C"]


def test_blank_arrival_is_filled_from_departure(tmp_path: Path) -> None:
    feed = _write_feed(
        tmp_path / "blank",
        **{
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,08:00:00,08:00:00,A,1\n"
                "t1,,08:11:00,B,2\n"
                "t1,08:20:00,,C,3\n"
            )
        },
    )
    timetable = load_timetable(feed)
    assert timetable.arrival_at(0, 0, 1) == 8 * 3600 + 11 * 60
    assert timetable.departure_at(0, 0, 2) == 8 * 3600 + 20 * 60


def test_trip_with_no_usable_times_is_dropped(tmp_path: Path) -> None:
    feed = _write_feed(
        tmp_path / "unusable",
        **{
            "trips.txt": "route_id,service_id,trip_id\nR1,WD,t1\nR1,WD,t2\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,08:00:00,08:00:00,A,1\n"
                "t1,08:10:00,08:11:00,B,2\n"
                "t2,,,A,1\n"
                "t2,,,B,2\n"
            ),
        },
    )
    stats = LoadStats()
    timetable = load_timetable(feed, stats=stats)
    assert stats.trips_dropped == 1
    assert timetable.n_trips == 1


def test_transfer_type_3_is_not_a_footpath(tmp_path: Path) -> None:
    feed = _write_feed(
        tmp_path / "transfers",
        **{
            "transfers.txt": (
                "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n"
                "A,B,2,120\n"
                "B,C,3,\n"
            )
        },
    )
    timetable = load_timetable(feed)
    assert dict(timetable.footpaths_from(timetable.stop_index["A"])) == {
        timetable.stop_index["B"]: 120
    }
    assert dict(timetable.footpaths_from(timetable.stop_index["B"])) == {}


def test_split_overtaking_separates_express_from_local() -> None:
    raw = RawPattern(
        route_id="MIXED",
        stops=["A", "B", "C"],
        trip_ids=["local", "express"],
        arrivals=[[0, 1200, 2400], [300, 600, 900]],
        departures=[[0, 1200, 2400], [300, 600, 900]],
    )
    pieces = split_overtaking(raw)
    assert len(pieces) == 2
    assert {tuple(p.trip_ids) for p in pieces} == {("local",), ("express",)}
    assert [p.stops for p in pieces] == [["A", "B", "C"], ["A", "B", "C"]]


def test_split_overtaking_leaves_a_clean_pattern_alone() -> None:
    raw = RawPattern(
        route_id="CLEAN",
        stops=["A", "B"],
        trip_ids=["t1", "t2"],
        arrivals=[[0, 600], [300, 900]],
        departures=[[0, 600], [300, 900]],
    )
    assert split_overtaking(raw) == [raw]


def test_overtaking_feed_loads_by_splitting(tmp_path: Path) -> None:
    """An express overtaking a local must not break the load."""
    feed = _write_feed(
        tmp_path / "express",
        **{
            "trips.txt": "route_id,service_id,trip_id\nR1,WD,local\nR1,WD,express\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "local,08:00:00,08:00:00,A,1\n"
                "local,08:30:00,08:30:00,B,2\n"
                "local,09:00:00,09:00:00,C,3\n"
                "express,08:05:00,08:05:00,A,1\n"
                "express,08:12:00,08:12:00,B,2\n"
                "express,08:20:00,08:20:00,C,3\n"
            ),
        },
    )
    stats = LoadStats()
    timetable = load_timetable(feed, stats=stats)
    assert stats.patterns_split == 1
    assert timetable.n_patterns == 2
    assert timetable.n_trips == 2
