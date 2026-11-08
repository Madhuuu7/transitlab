"""Re-encode a built :class:`Timetable` for a browser to load and route over.

Why this exists
---------------
The router is 400 lines over flat integer arrays. Nothing in it needs a
server: given the same arrays, the same rounds produce the same journeys in
any language with typed arrays. So the deployed console does not call an API
at all -- it downloads the timetable once and runs RAPTOR in the tab.

That turns the deployment problem inside out. :mod:`transitlab.gtfs.snapshot`
exists because ingest cannot fit on a free instance; this module exists
because *nothing* needs to fit on an instance. What it costs instead is
download, and a snapshot is the wrong shape to send over a network:

=========================  =========  =========
array                      snapshot   bundle
=========================  =========  =========
``departures`` (int32)       3.32 MB    0.77 MB
``arrivals`` (int32)         3.32 MB    omitted
everything else              0.85 MB    0.85 MB
=========================  =========  =========

Both savings come from looking at the feed rather than assuming things about
it, and both are **checked at build time, not asserted**:

1. **The feed records no dwell time.** In the BMTC feed ``arrivals`` and
   ``departures`` are equal at all 1,540,565 stop-times -- the agency
   publishes one time per stop, so half the stop-time data is a copy of the
   other half. :func:`encode` compares the two arrays and sets ``dwell``;
   when they differ, both tables ship. A feed with real dwell times pays for
   them and nothing is silently dropped.

2. **Times within a trip fit in 16 bits.** Absolute GTFS seconds need int32
   because the service day runs past 24:00, but a stop time measured from its
   own trip's first departure is at most a few hours -- 44,999 seconds here,
   and the format refuses anything above 65,535 rather than wrapping. Storing
   the offsets as uint16 beside one int32 per trip halves the bytes and, far
   more importantly, makes them compress: neighbouring offsets are close
   numbers, and gzip finds that where it cannot find structure in absolute
   timestamps spread over a day.

The decoder rebuilds the full int32 tables in memory before the router sees
them, so the encoding is invisible above this layer and the browser's hot
loop runs the same arithmetic as the server's.

Format
------
Three files, because they have three different lifetimes in a browser cache
and two of them are text:

``manifest.json``
    Self-describing: version, feed name, counts, the time encoding, and for
    every array its dtype, length, byte offset and byte length. A decoder
    reads the manifest and needs no compiled-in knowledge of the layout.
``timetable.bin``
    The arrays, concatenated in manifest order, each padded to an 8-byte
    boundary -- a ``Float64Array`` over a buffer rejects a misaligned offset,
    and discovering that in a browser console is worse than paying 7 bytes.
``strings.json``
    Stop ids and names, pattern route ids, trip ids. Text, which belongs in
    text: as JSON they compress to a fraction of what they would take padded
    into a numeric blob, and the router never touches them.

:func:`write` gzips all three. A static host may or may not compress an
unknown binary type on the way out, and the size of the download should not
depend on which; the console decompresses with ``DecompressionStream`` and
gets the same bytes either way.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .timetable import Stop, Timetable

#: Bumped whenever the meaning or layout of the bundle changes. The console
#: refuses a version it does not know rather than routing on a guess.
BUNDLE_VERSION = 2

#: Offsets are stored as uint16 only while every one of them fits.
U16_MAX = 65_535

_DTYPES: dict[str, np.dtype] = {
    "i32": np.dtype("<i4"),
    "u16": np.dtype("<u2"),
    "f64": np.dtype("<f8"),
}

#: Every array starts on an 8-byte boundary; see the module docstring.
_ALIGN = 8


class BundleError(ValueError):
    """Raised when a timetable cannot be bundled, or a bundle trusted."""


@dataclass(frozen=True)
class Bundle:
    """The three files, in memory."""

    manifest: dict
    binary: bytes
    strings: dict


# ----------------------------------------------------------------------
# encode
# ----------------------------------------------------------------------
def _trip_first_departures(timetable: Timetable) -> np.ndarray:
    """The first departure of every trip, in global trip order.

    This is the base each stop time is measured from. It is per *trip* rather
    than per pattern because the trips of one pattern are spread across the
    whole day: measuring from the pattern's earliest trip would put the last
    trip's offsets back above 16 bits, which is the whole point of this.
    """
    first = np.empty(timetable.n_trips, dtype=np.int32)
    for pattern in range(timetable.n_patterns):
        length = timetable.pattern_length(pattern)
        trips = timetable.trip_count(pattern)
        base = int(timetable.stop_time_offset[pattern])
        block = timetable.departures[base : base + trips * length].reshape(trips, length)
        lo = int(timetable.pattern_trip_offset[pattern])
        first[lo : lo + trips] = block[:, 0]
    return first


def _expand_base(
    trip_first: np.ndarray,
    pattern_stop_offset: np.ndarray,
    pattern_trip_offset: np.ndarray,
    stop_time_offset: np.ndarray,
    n_stop_times: int,
) -> np.ndarray:
    """One base departure per stop-time, expanded from one per trip."""
    out = np.zeros(n_stop_times, dtype=np.int64)
    n_patterns = len(pattern_stop_offset) - 1
    for pattern in range(n_patterns):
        length = int(pattern_stop_offset[pattern + 1] - pattern_stop_offset[pattern])
        trips = int(pattern_trip_offset[pattern + 1] - pattern_trip_offset[pattern])
        base = int(stop_time_offset[pattern])
        lo = int(pattern_trip_offset[pattern])
        out[base : base + trips * length] = np.repeat(
            trip_first[lo : lo + trips].astype(np.int64), length
        )
    return out


def encode(timetable: Timetable, feed_name: str) -> Bundle:
    """Encode ``timetable`` into the three bundle files."""
    if timetable.n_patterns == 0:
        raise BundleError("cannot bundle a timetable with no patterns")

    # stop_time_offset is int64 in the snapshot because it indexes a table
    # that could in principle exceed 2**31 entries. No real feed is close --
    # Bengaluru is 1.5 million -- and a browser has no cheap int64 array, so
    # it narrows here, after a check rather than on a hope.
    if int(timetable.stop_time_offset[-1]) > np.iinfo(np.int32).max:
        raise BundleError("stop-time table is too large to index with int32")

    dwell = not np.array_equal(timetable.arrivals, timetable.departures)

    trip_first = _trip_first_departures(timetable)
    base = _expand_base(
        trip_first,
        timetable.pattern_stop_offset,
        timetable.pattern_trip_offset,
        timetable.stop_time_offset,
        timetable.n_stop_times,
    )
    dep_delta = timetable.departures.astype(np.int64) - base
    arr_delta = timetable.arrivals.astype(np.int64) - base

    candidates = np.concatenate([dep_delta, arr_delta]) if dwell else dep_delta
    fits_u16 = bool(
        candidates.size and candidates.min() >= 0 and candidates.max() <= U16_MAX
    )
    encoding = "u16-delta" if fits_u16 else "i32-absolute"

    arrays: list[tuple[str, str, np.ndarray]] = [
        ("coords", "f64", timetable.coords.ravel()),
        ("pattern_stop_offset", "i32", timetable.pattern_stop_offset),
        ("pattern_stops", "i32", timetable.pattern_stops),
        ("pattern_trip_offset", "i32", timetable.pattern_trip_offset),
        ("stop_time_offset", "i32", timetable.stop_time_offset),
        ("stop_pattern_offset", "i32", timetable.stop_pattern_offset),
        ("stop_pattern_pattern", "i32", timetable.stop_pattern_pattern),
        ("stop_pattern_position", "i32", timetable.stop_pattern_position),
        ("footpath_offset", "i32", timetable.footpath_offset),
        ("footpath_target", "i32", timetable.footpath_target),
        ("footpath_time", "i32", timetable.footpath_time),
    ]

    if encoding == "u16-delta":
        arrays.append(("trip_first_departure", "i32", trip_first))
        arrays.append(("departure_offset", "u16", dep_delta))
        if dwell:
            arrays.append(("arrival_offset", "u16", arr_delta))
    else:
        arrays.append(("departures", "i32", timetable.departures))
        if dwell:
            arrays.append(("arrivals", "i32", timetable.arrivals))

    chunks: list[bytes] = []
    specs: list[dict] = []
    offset = 0
    for name, dtype, array in arrays:
        payload = np.ascontiguousarray(array, dtype=_DTYPES[dtype]).tobytes()
        specs.append(
            {
                "name": name,
                "dtype": dtype,
                "length": int(np.size(array)),
                "offset": offset,
                "bytes": len(payload),
            }
        )
        chunks.append(payload)
        offset += len(payload)
        pad = -offset % _ALIGN
        if pad:
            chunks.append(b"\x00" * pad)
            offset += pad

    manifest = {
        "version": BUNDLE_VERSION,
        "feed": feed_name,
        "encoding": encoding,
        "dwell": dwell,
        "counts": {
            "stops": timetable.n_stops,
            "patterns": timetable.n_patterns,
            "trips": timetable.n_trips,
            "stop_times": timetable.n_stop_times,
            "footpaths": int(timetable.footpath_target.size),
        },
        "arrays": specs,
    }
    strings = {
        "stop_ids": [s.stop_id for s in timetable.stops],
        "stop_names": [s.name for s in timetable.stops],
        "pattern_route_id": list(timetable.pattern_route_id),
        "pattern_route_name": list(timetable.pattern_route_name),
        "trip_ids": list(timetable.trip_ids),
    }
    return Bundle(manifest=manifest, binary=b"".join(chunks), strings=strings)


# ----------------------------------------------------------------------
# decode -- the reference the TypeScript decoder is checked against
# ----------------------------------------------------------------------
def decode(bundle: Bundle) -> Timetable:
    """Rebuild a :class:`Timetable` from a bundle.

    The console has its own decoder in TypeScript; this one exists so the
    format can be tested without a browser, and so "the bundle still means
    the timetable it came from" is a property the Python suite can assert.
    """
    manifest = bundle.manifest
    version = manifest.get("version")
    if version != BUNDLE_VERSION:
        raise BundleError(
            f"bundle is version {version}, this build reads {BUNDLE_VERSION}"
        )

    raw: dict[str, np.ndarray] = {}
    for spec in manifest["arrays"]:
        dtype = _DTYPES[spec["dtype"]]
        start = int(spec["offset"])
        length = int(spec["length"])
        if start + length * dtype.itemsize > len(bundle.binary):
            raise BundleError(f"{spec['name']} runs past the end of the bundle")
        raw[spec["name"]] = np.frombuffer(
            bundle.binary, dtype=dtype, count=length, offset=start
        )

    encoding = manifest["encoding"]
    dwell = bool(manifest["dwell"])
    counts = manifest["counts"]

    if encoding == "u16-delta":
        base = _expand_base(
            raw["trip_first_departure"],
            raw["pattern_stop_offset"],
            raw["pattern_trip_offset"],
            raw["stop_time_offset"],
            int(counts["stop_times"]),
        )
        departures = (base + raw["departure_offset"].astype(np.int64)).astype(np.int32)
        arrivals = (
            (base + raw["arrival_offset"].astype(np.int64)).astype(np.int32)
            if dwell
            else departures
        )
    elif encoding == "i32-absolute":
        departures = raw["departures"].astype(np.int32)
        arrivals = raw["arrivals"].astype(np.int32) if dwell else departures
    else:
        raise BundleError(f"unknown time encoding {encoding!r}")

    strings = bundle.strings
    coords = np.ascontiguousarray(raw["coords"], dtype=np.float64).reshape(-1, 2)
    stops = tuple(
        Stop(stop_id, name, float(coords[i, 0]), float(coords[i, 1]))
        for i, (stop_id, name) in enumerate(
            zip(strings["stop_ids"], strings["stop_names"])
        )
    )
    return Timetable(
        stops=stops,
        stop_index={s.stop_id: i for i, s in enumerate(stops)},
        coords=coords,
        pattern_route_id=tuple(strings["pattern_route_id"]),
        pattern_route_name=tuple(strings["pattern_route_name"]),
        pattern_stop_offset=np.ascontiguousarray(raw["pattern_stop_offset"], np.int32),
        pattern_stops=np.ascontiguousarray(raw["pattern_stops"], np.int32),
        pattern_trip_offset=np.ascontiguousarray(raw["pattern_trip_offset"], np.int32),
        trip_ids=tuple(strings["trip_ids"]),
        stop_time_offset=np.ascontiguousarray(raw["stop_time_offset"], np.int64),
        arrivals=np.ascontiguousarray(arrivals, np.int32),
        departures=np.ascontiguousarray(departures, np.int32),
        stop_pattern_offset=np.ascontiguousarray(raw["stop_pattern_offset"], np.int32),
        stop_pattern_pattern=np.ascontiguousarray(raw["stop_pattern_pattern"], np.int32),
        stop_pattern_position=np.ascontiguousarray(
            raw["stop_pattern_position"], np.int32
        ),
        footpath_offset=np.ascontiguousarray(raw["footpath_offset"], np.int32),
        footpath_target=np.ascontiguousarray(raw["footpath_target"], np.int32),
        footpath_time=np.ascontiguousarray(raw["footpath_time"], np.int32),
    )


# ----------------------------------------------------------------------
# files
# ----------------------------------------------------------------------
def _gzip(payload: bytes) -> bytes:
    # mtime=0 so rebuilding the same timetable produces the same bytes, and a
    # diff in the repository means the feed changed rather than the clock.
    return gzip.compress(payload, compresslevel=9, mtime=0)


def write(bundle: Bundle, directory: str | Path) -> dict[str, int]:
    """Write the gzipped bundle into ``directory``; return the file sizes.

    The two large files are compressed first and their sizes recorded in the
    manifest, so the console can state its own download size without
    measuring it. Measuring it is the thing that does not work: a host that
    sets ``Content-Encoding: gzip`` on a ``.gz`` file has the browser inflate
    it before any code can see it, and the figure on the page then changes
    with the host rather than with the feed. These numbers are a property of
    the build.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    sizes: dict[str, int] = {}
    for name, payload in (
        ("timetable.bin", bundle.binary),
        ("strings.json", json.dumps(bundle.strings, separators=(",", ":")).encode()),
    ):
        compressed = _gzip(payload)
        (out / f"{name}.gz").write_bytes(compressed)
        sizes[name] = len(compressed)

    manifest = dict(bundle.manifest)
    manifest["download"] = dict(sizes)
    encoded = _gzip(json.dumps(manifest, separators=(",", ":")).encode())
    (out / "manifest.json.gz").write_bytes(encoded)
    sizes["manifest.json"] = len(encoded)
    return sizes


def read(directory: str | Path) -> Bundle:
    """Read a bundle written by :func:`write`."""
    source = Path(directory)
    try:
        manifest = json.loads(
            gzip.decompress((source / "manifest.json.gz").read_bytes())
        )
        binary = gzip.decompress((source / "timetable.bin.gz").read_bytes())
        strings = json.loads(gzip.decompress((source / "strings.json.gz").read_bytes()))
    except FileNotFoundError as exc:
        raise BundleError(f"{source} is not a bundle directory: {exc}") from None
    return Bundle(manifest=manifest, binary=binary, strings=strings)
