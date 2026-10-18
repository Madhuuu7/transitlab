"""Save and load a built :class:`Timetable` as a single file.

Why this exists
---------------
Ingest and query have wildly different resource profiles, and conflating them
makes the cheap one pay for the expensive one.

Building the Bengaluru timetable costs about 15 seconds and peaks near 1.6 GB
of resident memory: DuckDB's scan buffers, plus a Python list per trip while
patterns are grouped, over 1.54 million stop-times. **The result of all that
is 15.4 MB of flat arrays.** Everything else was scaffolding, and most of it
is never returned to the OS, so a server that ingests on boot carries the
peak for its whole life.

So ingest becomes a build step. :func:`save` writes the finished arrays;
:func:`load` maps them back in a fraction of a second with no DuckDB, no CSV
parsing and no intermediate Python objects. A 512 MB free-tier instance can
serve a city it could not possibly have ingested.

Format
------
A compressed ``.npz`` holding every array of the timetable plus one JSON
member for the strings -- stop ids and names, pattern route ids, trip ids --
which do not belong in a numeric archive and compress better as text.

``stop_index`` is not stored; it is rebuilt on load, because a dict that can
be derived from a list it must stay consistent with is a bug waiting to
happen. The derived scalar caches (``_length`` and friends) are likewise
rebuilt by ``Timetable.__post_init__``.

The archive carries a format version. A snapshot built by a different layout
of the CSR arrays would otherwise load successfully and route wrongly, which
is the worst available outcome; instead it refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .timetable import Stop, Timetable

#: Bumped whenever the meaning or layout of the stored arrays changes.
SNAPSHOT_VERSION = 1

#: Arrays stored verbatim, with the dtype each must come back as. The dtypes
#: are restated here rather than trusted from the file so that a snapshot
#: written by an older build cannot quietly widen an index array.
_ARRAYS: dict[str, type] = {
    "coords": np.float64,
    "pattern_stop_offset": np.int32,
    "pattern_stops": np.int32,
    "pattern_trip_offset": np.int32,
    "stop_time_offset": np.int64,
    "arrivals": np.int32,
    "departures": np.int32,
    "stop_pattern_offset": np.int32,
    "stop_pattern_pattern": np.int32,
    "stop_pattern_position": np.int32,
    "footpath_offset": np.int32,
    "footpath_target": np.int32,
    "footpath_time": np.int32,
}


class SnapshotError(ValueError):
    """Raised when a snapshot cannot be trusted to route correctly."""


def save(timetable: Timetable, path: str | Path) -> Path:
    """Write ``timetable`` to ``path`` as a compressed archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.dumps(
        {
            "version": SNAPSHOT_VERSION,
            "stop_ids": [s.stop_id for s in timetable.stops],
            "stop_names": [s.name for s in timetable.stops],
            "pattern_route_id": list(timetable.pattern_route_id),
            "trip_ids": list(timetable.trip_ids),
        },
        separators=(",", ":"),
    )

    arrays = {name: getattr(timetable, name) for name in _ARRAYS}
    # savez_compressed only takes arrays, so the JSON rides along as a 0-d
    # unicode array rather than in a second file beside the archive.
    np.savez_compressed(destination, meta=np.array(metadata), **arrays)

    # np.savez_compressed appends .npz when the name lacks it; return the path
    # that actually exists so callers can report something true.
    if not destination.exists() and destination.with_suffix(".npz").exists():
        return destination.with_suffix(".npz")
    return destination


def load(path: str | Path) -> Timetable:
    """Read a snapshot written by :func:`save`."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        try:
            metadata = json.loads(str(archive["meta"]))
        except KeyError:
            raise SnapshotError(f"{source.name} is not a transitlab snapshot") from None

        version = metadata.get("version")
        if version != SNAPSHOT_VERSION:
            raise SnapshotError(
                f"{source.name} is snapshot version {version}, "
                f"this build reads version {SNAPSHOT_VERSION}; rebuild it"
            )

        missing = [name for name in _ARRAYS if name not in archive]
        if missing:
            raise SnapshotError(f"{source.name} is missing {', '.join(missing)}")

        arrays = {
            name: np.ascontiguousarray(archive[name], dtype=dtype)
            for name, dtype in _ARRAYS.items()
        }

    stop_ids = metadata["stop_ids"]
    stop_names = metadata["stop_names"]
    coords = arrays["coords"]
    if len(stop_ids) != len(stop_names) or len(stop_ids) != coords.shape[0]:
        raise SnapshotError(f"{source.name} has inconsistent stop metadata")

    stops = tuple(
        Stop(stop_id, name, float(coords[i, 0]), float(coords[i, 1]))
        for i, (stop_id, name) in enumerate(zip(stop_ids, stop_names))
    )

    return Timetable(
        stops=stops,
        stop_index={s.stop_id: i for i, s in enumerate(stops)},
        pattern_route_id=tuple(metadata["pattern_route_id"]),
        trip_ids=tuple(metadata["trip_ids"]),
        **arrays,
    )
