"""GTFS time handling.

GTFS stores stop times as ``HH:MM:SS`` relative to *noon minus 12 hours* on the
service day, and the hour field is deliberately allowed to exceed 24. A trip
that leaves at 23:50 and arrives at 00:20 the next calendar day is written as
``24:20:00``, not ``00:20:00`` -- that is how a feed keeps a single trip on a
single service day instead of splitting it across two.

We therefore store seconds since the start of the service day in an int32 and
never wrap at 86400. Wrapping would silently turn a 30-minute trip into a
23.5-hour one, which is the classic GTFS parsing bug.
"""

from __future__ import annotations

# Sentinel for "unreachable" / "no time". int32 max is 2147483647; we leave
# headroom so that `arrival + transfer_time` cannot overflow during routing.
NEVER: int = 2_000_000_000

SECONDS_PER_DAY: int = 86_400


def parse_gtfs_time(value: str) -> int:
    """Parse a GTFS ``HH:MM:SS`` string into seconds since service-day start.

    Hours above 23 are valid and preserved. Single-digit hours ("7:05:00") are
    accepted because real feeds contain them despite the spec.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty GTFS time")

    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"malformed GTFS time: {value!r}")

    hours, minutes, seconds = (int(p) for p in parts)
    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError(f"minutes/seconds out of range: {value!r}")
    if hours < 0:
        raise ValueError(f"negative hour: {value!r}")

    return hours * 3600 + minutes * 60 + seconds


def format_gtfs_time(seconds: int) -> str:
    """Inverse of :func:`parse_gtfs_time`, keeping hours >= 24 intact."""
    if seconds >= NEVER:
        return "--:--:--"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
