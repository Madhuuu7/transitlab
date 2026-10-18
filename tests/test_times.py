from __future__ import annotations

import pytest

from transitlab.gtfs.times import NEVER, format_gtfs_time, parse_gtfs_time


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("00:00:00", 0),
        ("06:30:00", 23_400),
        ("7:05:00", 25_500),  # single-digit hour, out of spec but common
        ("23:59:59", 86_399),
        ("24:05:00", 86_700),  # past midnight on the same service day
        ("27:30:00", 99_000),
    ],
)
def test_parse(text: str, seconds: int) -> None:
    assert parse_gtfs_time(text) == seconds


def test_hours_past_midnight_do_not_wrap() -> None:
    """The bug this module exists to prevent."""
    departure = parse_gtfs_time("23:50:00")
    arrival = parse_gtfs_time("24:05:00")
    assert arrival - departure == 15 * 60


def test_round_trip() -> None:
    for text in ("00:00:00", "12:34:56", "24:05:00", "30:00:00"):
        assert format_gtfs_time(parse_gtfs_time(text)) == text


def test_never_formats_as_unreachable() -> None:
    assert format_gtfs_time(NEVER) == "--:--:--"


@pytest.mark.parametrize("bad", ["", "   ", "12:34", "12:34:56:78", "12:60:00", "12:00:61"])
def test_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_gtfs_time(bad)
