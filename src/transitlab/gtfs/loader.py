"""Load a GTFS feed (zip or extracted directory) into a :class:`Timetable`.

Why DuckDB and not ``csv``/pandas
---------------------------------
``stop_times.txt`` is the only file that is ever large -- millions of rows for
a city feed, and 90% of ingest time. The work done to it is a sort and a
group-by, which is exactly what a columnar engine is for: DuckDB reads the CSV
with a parallel vectorised scanner and does the grouping in C++. The time
parsing happens in SQL too (``split_part`` over the ``HH:MM:SS`` field), so the
strings never cross into Python.

Pattern extraction
------------------
RAPTOR's "route" is **not** a GTFS ``route_id``; it is a maximal set of trips
that visit the identical stop sequence. One ``route_id`` routinely holds
several: a full run and a short turn, or the two directions. Grouping by
``route_id`` instead would break the algorithm's assumption that any trip of a
route can be boarded at any of its stops.

The sequence is recovered in SQL with
``string_agg(stop_id, '>' ORDER BY stop_sequence)`` per trip, and trips sharing
a ``(route_id, sequence)`` key become one pattern.

Feed quirks handled here
------------------------
* ``stop_sequence`` need only be increasing, not consecutive, so positions are
  derived from the sort order rather than from the field itself.
* A blank ``arrival_time``/``departure_time`` is filled from its partner;
  trips with neither are dropped rather than silently interpolated.
* Station rows (``location_type`` 1-4) are not boardable and are skipped.
* Trips whose stop sequence collapses to fewer than two stops are dropped.
* Patterns whose trips overtake each other are split until each piece is
  totally ordered -- see :func:`split_overtaking`.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import duckdb

from .footpaths import generate_footpaths
from .timetable import (
    MAX_FOOTPATH_SECONDS,
    RawPattern,
    Stop,
    Timetable,
    build_timetable,
)

# Only these members are read; anything else in the archive is ignored.
GTFS_MEMBERS = (
    "agency.txt",
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "transfers.txt",
    "frequencies.txt",
)

# Transfer types 0/1/2 are usable; 3 means "transfer not possible" and must be
# excluded rather than treated as a zero-cost walk.
USABLE_TRANSFER_TYPES = ("", "0", "1", "2")

_SECONDS_MACRO = """
CREATE OR REPLACE TEMP MACRO gtfs_seconds(t) AS
    CAST(split_part(t, ':', 1) AS INTEGER) * 3600
  + CAST(split_part(t, ':', 2) AS INTEGER) * 60
  + CAST(split_part(t, ':', 3) AS INTEGER);
"""


class FeedError(ValueError):
    """The archive is not a usable GTFS feed."""


@dataclass
class LoadStats:
    """What the loader did, for logging and for the ingest benchmark."""

    stops: int = 0
    trips_read: int = 0
    trips_dropped: int = 0
    patterns: int = 0
    patterns_split: int = 0
    stop_times: int = 0
    footpaths_declared: int = 0
    footpaths_generated: int = 0

    def __str__(self) -> str:
        walking = f"{self.footpaths_declared} declared footpaths"
        if self.footpaths_generated:
            walking += f" + {self.footpaths_generated} generated from geography"
        return (
            f"{self.stops} stops, {self.trips_read} trips "
            f"({self.trips_dropped} dropped), {self.patterns} patterns "
            f"({self.patterns_split} from splitting overtaken ones), "
            f"{self.stop_times} stop-times, "
            f"{walking}"
        )


def load_timetable(
    source: str | Path,
    *,
    service_ids: Iterable[str] | None = None,
    date: str | None = None,
    max_footpath_seconds: int = MAX_FOOTPATH_SECONDS,
    walk_radius_m: float | None = None,
    stats: LoadStats | None = None,
) -> Timetable:
    """Read a GTFS zip or directory and return the flat timetable.

    ``date`` ("YYYYMMDD") selects the services running that day from
    ``calendar.txt`` plus ``calendar_dates.txt`` exceptions. ``service_ids``
    overrides that with an explicit set. With neither, every trip in the feed
    is loaded, which is what the tests and benchmarks want but is wrong for a
    real query -- a feed holds weekday, Saturday and Sunday service at once.

    ``walk_radius_m`` adds walking transfers between stops within that many
    straight-line metres, on top of whatever ``transfers.txt`` declares. It is
    opt-in and off by default, because inventing edges that an agency did not
    publish is a modelling decision the caller should make knowingly -- but it
    is close to mandatory on a feed that declares no transfers at all, where
    the router is otherwise forbidden from crossing a road. See
    :mod:`transitlab.gtfs.footpaths`.
    """
    stats = stats if stats is not None else LoadStats()

    with _feed_directory(source) as directory:
        con = duckdb.connect()
        try:
            con.execute(_SECONDS_MACRO)
            _register(con, directory)

            selected = _resolve_services(con, service_ids=service_ids, date=date)
            stops = _read_stops(con)
            patterns = _read_patterns(con, selected, stats)
            footpaths = _read_transfers(con, stats)
        finally:
            con.close()

    if walk_radius_m:
        generated = generate_footpaths(
            stops,
            radius_m=walk_radius_m,
            exclude=[(a, b) for a, b, _ in footpaths],
        )
        stats.footpaths_generated = len(generated)
        footpaths = [*footpaths, *generated]

    stop_ids = {s.stop_id for s in stops}
    patterns = [p for p in patterns if all(s in stop_ids for s in p.stops)]

    split: list[RawPattern] = []
    for raw in patterns:
        pieces = split_overtaking(raw)
        stats.patterns_split += len(pieces) - 1
        split.extend(pieces)

    stats.stops = len(stops)
    stats.patterns = len(split)
    stats.stop_times = sum(len(p.stops) * len(p.trip_ids) for p in split)

    return build_timetable(
        stops, split, footpaths, max_footpath_seconds=max_footpath_seconds
    )


# ----------------------------------------------------------------------
# archive handling
# ----------------------------------------------------------------------
class _feed_directory:
    """Context manager yielding a directory of GTFS ``.txt`` files.

    A zip is extracted member by member into a fresh temp directory, writing
    only to ``basename``. Zip entries can carry ``../`` or absolute paths, and
    ``extractall`` on an untrusted archive is a path-traversal write.
    """

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)
        self._temp: Path | None = None

    def __enter__(self) -> Path:
        if self.source.is_dir():
            return self.source
        if not self.source.is_file():
            raise FeedError(f"no such feed: {self.source}")

        self._temp = Path(tempfile.mkdtemp(prefix="transitlab-feed-"))
        try:
            with zipfile.ZipFile(self.source) as zf:
                names = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
                wanted = [m for m in GTFS_MEMBERS if m in names]
                if "stop_times.txt" not in wanted or "trips.txt" not in wanted:
                    raise FeedError(
                        f"{self.source.name} lacks trips.txt/stop_times.txt; "
                        "not a GTFS feed"
                    )
                for member in wanted:
                    with zf.open(names[member]) as src:
                        (self._temp / member).write_bytes(src.read())
        except zipfile.BadZipFile as exc:
            self.__exit__(None, None, None)
            raise FeedError(f"{self.source} is not a zip archive") from exc
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self._temp

    def __exit__(self, *_exc: object) -> None:
        if self._temp is not None:
            shutil.rmtree(self._temp, ignore_errors=True)
            self._temp = None


def _register(con: duckdb.DuckDBPyConnection, directory: Path) -> None:
    """Expose each present GTFS file as an all-varchar view.

    ``all_varchar`` is deliberate: ``stop_id`` "007" must not become the
    integer 7, and times like "24:20:00" are not a time type anywhere.
    """
    present = []
    for member in GTFS_MEMBERS:
        path = directory / member
        if not path.is_file():
            continue
        view = member[:-4]
        literal = str(path).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW {view} AS "
            f"SELECT * FROM read_csv('{literal}', all_varchar = true, "
            "header = true, nullstr = '', ignore_errors = false)"
        )
        present.append(view)
    for required in ("trips", "stop_times", "stops"):
        if required not in present:
            raise FeedError(f"feed is missing {required}.txt")


def _has_view(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    rows = con.execute(
        "SELECT 1 FROM duckdb_views() WHERE view_name = ? LIMIT 1", [name]
    ).fetchall()
    return bool(rows)


def _columns(con: duckdb.DuckDBPyConnection, view: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info('{view}')").fetchall()}


# ----------------------------------------------------------------------
# service selection
# ----------------------------------------------------------------------
_WEEKDAY_COLUMNS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _resolve_services(
    con: duckdb.DuckDBPyConnection,
    *,
    service_ids: Iterable[str] | None,
    date: str | None,
) -> set[str] | None:
    if service_ids is not None:
        return set(service_ids)
    if date is None:
        return None
    return services_on_date(con, date)


def services_on_date(con: duckdb.DuckDBPyConnection, date: str) -> set[str]:
    """Service ids running on ``date`` ("YYYYMMDD").

    ``calendar.txt`` gives a weekly pattern within a date range;
    ``calendar_dates.txt`` then adds (exception_type 1) or removes
    (exception_type 2) individual days. A feed may ship either file alone --
    some agencies express everything as explicit dates -- so both are
    optional.
    """
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"date must be YYYYMMDD, got {date!r}")

    active: set[str] = set()
    if _has_view(con, "calendar"):
        import datetime as _dt

        weekday = _dt.date(int(date[:4]), int(date[4:6]), int(date[6:])).weekday()
        column = _WEEKDAY_COLUMNS[weekday]
        rows = con.execute(
            f"""
            SELECT service_id FROM calendar
            WHERE {column} = '1'
              AND start_date <= ? AND end_date >= ?
            """,
            [date, date],
        ).fetchall()
        active.update(r[0] for r in rows)

    if _has_view(con, "calendar_dates"):
        rows = con.execute(
            "SELECT service_id, exception_type FROM calendar_dates WHERE date = ?",
            [date],
        ).fetchall()
        for service_id, exception in rows:
            if exception == "1":
                active.add(service_id)
            elif exception == "2":
                active.discard(service_id)

    return active


# ----------------------------------------------------------------------
# table readers
# ----------------------------------------------------------------------
def _read_stops(con: duckdb.DuckDBPyConnection) -> list[Stop]:
    columns = _columns(con, "stops")
    name = "stop_name" if "stop_name" in columns else "stop_id"
    location = (
        "COALESCE(location_type, '0')" if "location_type" in columns else "'0'"
    )
    lat = "stop_lat" if "stop_lat" in columns else "NULL"
    lon = "stop_lon" if "stop_lon" in columns else "NULL"

    rows = con.execute(
        f"""
        SELECT stop_id,
               COALESCE({name}, stop_id) AS name,
               TRY_CAST({lat} AS DOUBLE) AS lat,
               TRY_CAST({lon} AS DOUBLE) AS lon
        FROM stops
        WHERE {location} IN ('', '0')
        ORDER BY stop_id
        """
    ).fetchall()
    if not rows:
        raise FeedError("stops.txt has no boardable stops")
    return [Stop(r[0], r[1], r[2] if r[2] is not None else 0.0, r[3] if r[3] is not None else 0.0) for r in rows]


def _read_patterns(
    con: duckdb.DuckDBPyConnection,
    service_ids: set[str] | None,
    stats: LoadStats,
) -> list[RawPattern]:
    """Group trips into patterns and return them with their stop times.

    One pass of SQL does the heavy lifting (parse, sort, group, key); Python
    only walks the already-ordered result and packs it into lists.
    """
    trip_columns = _columns(con, "trips")
    route_expr = "t.route_id" if "route_id" in trip_columns else "''"

    service_filter = ""
    params: list[object] = []
    if service_ids is not None:
        if not service_ids:
            raise FeedError("no services selected -- check the date or service_ids")
        if "service_id" in trip_columns:
            placeholders = ", ".join("?" for _ in service_ids)
            service_filter = f"WHERE t.service_id IN ({placeholders})"
            params = sorted(service_ids)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _st AS
        SELECT s.trip_id,
               {route_expr} AS route_id,
               CAST(s.stop_sequence AS INTEGER) AS seq,
               s.stop_id,
               gtfs_seconds(COALESCE(s.arrival_time, s.departure_time)) AS arr,
               gtfs_seconds(COALESCE(s.departure_time, s.arrival_time)) AS dep
        FROM stop_times s
        JOIN trips t USING (trip_id)
        {service_filter}
        """,
        params,
    )
    # Trips with an unparseable or missing time anywhere are unusable: a
    # missing intermediate time needs interpolation, which is a modelling
    # decision, not a parsing one, so they are dropped and counted.
    total_trips = con.execute("SELECT count(DISTINCT trip_id) FROM _st").fetchone()[0]
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _trip AS
        SELECT trip_id,
               any_value(route_id) AS route_id,
               string_agg(stop_id, '>' ORDER BY seq) AS pattern_key,
               count(*) AS n_stops,
               min(dep) AS first_dep,
               bool_or(arr IS NULL OR dep IS NULL) AS incomplete
        FROM _st
        GROUP BY trip_id
        """
    )
    usable = con.execute(
        "SELECT count(*) FROM _trip WHERE NOT incomplete AND n_stops >= 2"
    ).fetchone()[0]
    stats.trips_read = int(total_trips)
    stats.trips_dropped = int(total_trips) - int(usable)
    if not usable:
        # An empty timetable answers "unreachable" to every query, which looks
        # like a routing bug rather than a selection mistake. Fail here.
        raise FeedError(
            "no usable trips after service selection: "
            f"{total_trips} trips matched, all dropped for missing times or "
            "fewer than two stops"
            if total_trips
            else "no trips matched the selected services"
        )

    rows = con.execute(
        """
        SELECT p.route_id, p.pattern_key, s.trip_id, s.stop_id, s.arr, s.dep
        FROM _st s
        JOIN _trip p USING (trip_id)
        WHERE NOT p.incomplete AND p.n_stops >= 2
        ORDER BY p.route_id, p.pattern_key, p.first_dep, s.trip_id, s.seq
        """
    ).fetchall()

    # Rows arrive grouped by (route_id, pattern_key) and, inside a group, by
    # trip then stop_sequence, so one linear pass fills each pattern: a new
    # trip_id starts a new row of the stop-time table.
    grouped: dict[tuple[str, str], tuple[list[str], list[list[int]], list[list[int]]]] = {}
    for route_id, pattern_key, row_trip, _stop_id, arr, dep in rows:
        key = (route_id, pattern_key)
        bucket = grouped.get(key)
        if bucket is None:
            bucket = ([], [], [])
            grouped[key] = bucket
        trip_ids, arrivals, departures = bucket
        if not trip_ids or trip_ids[-1] != row_trip:
            trip_ids.append(row_trip)
            arrivals.append([])
            departures.append([])
        arrivals[-1].append(int(arr))
        departures[-1].append(int(dep))

    return [
        RawPattern(
            route_id=route_id,
            stops=pattern_key.split(">"),
            trip_ids=trip_ids,
            arrivals=arrivals,
            departures=departures,
        )
        for (route_id, pattern_key), (trip_ids, arrivals, departures) in grouped.items()
    ]


def _read_transfers(
    con: duckdb.DuckDBPyConnection, stats: LoadStats
) -> list[tuple[str, str, int]]:
    if not _has_view(con, "transfers"):
        return []
    columns = _columns(con, "transfers")
    if not {"from_stop_id", "to_stop_id"} <= columns:
        return []

    transfer_type = (
        "COALESCE(transfer_type, '0')" if "transfer_type" in columns else "'0'"
    )
    min_time = (
        "COALESCE(TRY_CAST(min_transfer_time AS INTEGER), 0)"
        if "min_transfer_time" in columns
        else "0"
    )
    placeholders = ", ".join("?" for _ in USABLE_TRANSFER_TYPES)
    rows = con.execute(
        f"""
        SELECT from_stop_id, to_stop_id, {min_time} AS secs
        FROM transfers
        WHERE {transfer_type} IN ({placeholders})
          AND from_stop_id IS NOT NULL AND to_stop_id IS NOT NULL
          AND from_stop_id <> to_stop_id
        """,
        list(USABLE_TRANSFER_TYPES),
    ).fetchall()
    stats.footpaths_declared = len(rows)
    return [(r[0], r[1], int(r[2])) for r in rows]


# ----------------------------------------------------------------------
# overtaking
# ----------------------------------------------------------------------
def split_overtaking(raw: RawPattern) -> list[RawPattern]:
    """Split a pattern until no piece contains trips that overtake.

    RAPTOR binary-searches the departure column of a pattern, so the trips
    must be ordered the same way at every stop. An express sharing a stop
    sequence with a local violates that. Splitting into bands preserves
    correctness at the cost of scanning one extra pattern at those stops --
    the standard fix, and cheaper than giving up the binary search.

    Greedy first-fit over trips sorted by first departure. A trip joins the
    first band whose last trip it never passes; ordering within a band is
    transitive, so comparing against that band's last trip is enough.
    """
    order = sorted(
        range(len(raw.trip_ids)), key=lambda t: (raw.departures[t][0], raw.trip_ids[t])
    )
    bands: list[list[int]] = []
    for t in order:
        for band in bands:
            previous = raw.departures[band[-1]]
            if all(a <= b for a, b in zip(previous, raw.departures[t])):
                band.append(t)
                break
        else:
            bands.append([t])

    if len(bands) == 1:
        return [raw]

    pieces: list[RawPattern] = []
    for index, band in enumerate(bands):
        pieces.append(
            RawPattern(
                route_id=raw.route_id if index == 0 else f"{raw.route_id}#{index}",
                stops=list(raw.stops),
                trip_ids=[raw.trip_ids[t] for t in band],
                arrivals=[list(raw.arrivals[t]) for t in band],
                departures=[list(raw.departures[t]) for t in band],
            )
        )
    return pieces
