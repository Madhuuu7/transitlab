"""The flat timetable RAPTOR runs on.

Everything here is a NumPy array in compressed-sparse-row (CSR) layout: one
``offset`` array of length ``n + 1`` plus one dense payload array. No per-stop
or per-trip Python objects survive the build. Two reasons:

1. **Locality.** RAPTOR's inner loop walks the stops of one pattern in order
   and reads one stop-time per stop. With a flat ``int32`` array those reads
   are sequential; with a list of ``StopTime`` objects each one is a pointer
   chase into a different cache line.
2. **Vectorisation.** Finding the earliest boardable trip becomes a
   ``np.searchsorted`` over a strided view of the departure array, with no
   copy and no Python loop over trips.

Index arithmetic
----------------
A pattern ``p`` has ``L = pattern_length(p)`` stops and ``T`` trips. Its
stop-times live in one contiguous block starting at ``stop_time_offset[p]``,
laid out trip-major::

    index = stop_time_offset[p] + trip * L + position

Trip-major (rather than position-major) is the right choice because the
stop-times of a single trip are read together when a journey is reconstructed,
while the column of departures for one position -- which is what the binary
search needs -- is still reachable as a stride-``L`` view costing nothing.

Non-overtaking
--------------
The binary search is only valid if the trips of a pattern are sorted by
departure time at *every* position simultaneously, i.e. no trip overtakes
another. Real feeds mostly satisfy this; express services sharing a stop
sequence with a local do not. Rather than return silently wrong journeys, the
builder sorts trips by their departure at the first stop and then verifies
every later position, raising :class:`OvertakingError` if the invariant fails.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

# Footpaths are walking links. Transitively closing them without a bound would
# connect every stop to every other through a chain of short hops, which is
# both wrong (nobody walks 3 km between buses) and quadratic in memory.
MAX_FOOTPATH_SECONDS: int = 900

#: Trip count at or below which :meth:`Timetable.earliest_trip` scans linearly
#: instead of binary-searching.
#:
#: Binary search wins on asymptotics and loses badly on constants at this size.
#: ``np.searchsorted`` over a strided view costs a slice object plus a NumPy
#: dispatch -- several microseconds -- regardless of how few elements it then
#: looks at, while a Python loop over a memoryview costs tens of nanoseconds
#: per element.
#:
#: The threshold matters because real feeds are shaped nothing like the
#: synthetic ones this was first benchmarked on. A generated feed gives every
#: pattern a regular headway and so dozens of trips; in the BMTC feed for
#: Bengaluru the **median pattern has 2 trips** and 74% have four or fewer,
#: because a pattern is a distinct stop sequence and most bus routes run a
#: handful of distinct variants. Binary search was the wrong trade on the data
#: the router actually sees, which the synthetic benchmark could not reveal.
LINEAR_SCAN_TRIPS: int = 24


def _int32_view(array: np.ndarray) -> memoryview:
    """A flat int32 memoryview over ``array``, for cheap scalar reads.

    The memoryview keeps the underlying buffer alive, so the array it is taken
    from does not need to be stored separately.
    """
    contiguous = np.ascontiguousarray(array, dtype=np.int32)
    return memoryview(contiguous).cast("B").cast("i")


class OvertakingError(ValueError):
    """Raised when a pattern's trips cannot be totally ordered by time."""


@dataclass(frozen=True)
class Stop:
    stop_id: str
    name: str
    lat: float
    lon: float


@dataclass
class RawPattern:
    """A pattern as the loader extracts it, before flattening.

    ``arrivals[t][i]`` and ``departures[t][i]`` are seconds since the start of
    the service day for trip ``t`` at position ``i``; see
    :mod:`transitlab.gtfs.times` for why they are not reduced modulo a day.
    """

    route_id: str
    stops: list[str]
    trip_ids: list[str]
    arrivals: list[list[int]]
    departures: list[list[int]]

    def __post_init__(self) -> None:
        if len(self.stops) < 2:
            raise ValueError(f"pattern {self.route_id} has fewer than 2 stops")
        if not self.trip_ids:
            raise ValueError(f"pattern {self.route_id} has no trips")
        width = len(self.stops)
        for name, table in (("arrivals", self.arrivals), ("departures", self.departures)):
            if len(table) != len(self.trip_ids):
                raise ValueError(f"{name} rows != trip count in {self.route_id}")
            for row in table:
                if len(row) != width:
                    raise ValueError(f"{name} row width != stop count in {self.route_id}")


@dataclass(frozen=True)
class Timetable:
    """Immutable flat timetable. Build it with :func:`build_timetable`."""

    stops: tuple[Stop, ...]
    stop_index: dict[str, int]
    coords: np.ndarray  # (S, 2) float64, lat/lon

    # --- patterns -------------------------------------------------------
    pattern_route_id: tuple[str, ...]
    pattern_stop_offset: np.ndarray  # (P + 1,) int32, into pattern_stops
    pattern_stops: np.ndarray  # (sum L_p,) int32, stop indices in order
    pattern_trip_offset: np.ndarray  # (P + 1,) int32, into trip_ids
    trip_ids: tuple[str, ...]
    stop_time_offset: np.ndarray  # (P + 1,) int64, into arrivals/departures
    arrivals: np.ndarray  # (sum T_p * L_p,) int32
    departures: np.ndarray  # same shape

    # --- reverse index: which patterns serve a stop, and where ----------
    stop_pattern_offset: np.ndarray  # (S + 1,) int32
    stop_pattern_pattern: np.ndarray  # int32
    stop_pattern_position: np.ndarray  # int32

    # --- footpaths, transitively closed, excluding the trivial self-link -
    footpath_offset: np.ndarray  # (S + 1,) int32
    footpath_target: np.ndarray  # int32
    footpath_time: np.ndarray  # int32

    # --- scalar read caches, derived; not part of the value -------------
    # The RAPTOR inner loop asks for a pattern's length, trip count and base
    # offset millions of times per query, and reads individual stop times
    # about as often. Every one of those from an int32 NumPy array pays a
    # scalar unbox; from a Python list or an int32 memoryview it is an
    # ordinary indexing operation an order of magnitude cheaper. The arrays
    # remain the source of truth -- these are built once per feed from them.
    _length: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _trips: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _base: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _dep: memoryview = field(init=False, repr=False, compare=False)
    _arr: memoryview = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        stop_offsets = self.pattern_stop_offset.tolist()
        trip_offsets = self.pattern_trip_offset.tolist()
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        set_("_length", tuple(
            stop_offsets[p + 1] - stop_offsets[p] for p in range(len(stop_offsets) - 1)
        ))
        set_("_trips", tuple(
            trip_offsets[p + 1] - trip_offsets[p] for p in range(len(trip_offsets) - 1)
        ))
        set_("_base", tuple(self.stop_time_offset.tolist()))
        # An int32 buffer already exposes format 'i', but a non-contiguous or
        # differently-typed array would not, and memoryview.cast refuses to go
        # directly between two non-byte formats -- hence the trip through 'B'.
        # Asking for contiguity keeps a hand-built Timetable from failing
        # somewhere less obvious than here.
        set_("_dep", _int32_view(self.departures))
        set_("_arr", _int32_view(self.arrivals))

    # ------------------------------------------------------------------
    # sizes
    # ------------------------------------------------------------------
    @property
    def n_stops(self) -> int:
        return len(self.stops)

    @property
    def n_patterns(self) -> int:
        return len(self.pattern_route_id)

    @property
    def n_trips(self) -> int:
        return len(self.trip_ids)

    @property
    def n_stop_times(self) -> int:
        return int(self.arrivals.size)

    def pattern_length(self, pattern: int) -> int:
        return self._length[pattern]

    def trip_count(self, pattern: int) -> int:
        return self._trips[pattern]

    # ------------------------------------------------------------------
    # views (no copies)
    # ------------------------------------------------------------------
    def stops_of(self, pattern: int) -> np.ndarray:
        lo = self.pattern_stop_offset[pattern]
        hi = self.pattern_stop_offset[pattern + 1]
        return self.pattern_stops[lo:hi]

    def position_of(self, pattern: int, stop: int) -> int:
        """First position at which ``stop`` occurs in ``pattern``, else -1."""
        where = np.flatnonzero(self.stops_of(pattern) == stop)
        return int(where[0]) if where.size else -1

    def departure_column(self, pattern: int, position: int) -> np.ndarray:
        """Departures of every trip of ``pattern`` at ``position``.

        A stride-``L`` view into :attr:`departures` -- no allocation. Sorted
        ascending because of the non-overtaking invariant, which is what makes
        the binary search in :meth:`earliest_trip` legal.
        """
        base = self._base[pattern] + position
        length = self._length[pattern]
        trips = self._trips[pattern]
        return self.departures[base : base + trips * length : length]

    def arrival_column(self, pattern: int, position: int) -> np.ndarray:
        base = self._base[pattern] + position
        length = self._length[pattern]
        trips = self._trips[pattern]
        return self.arrivals[base : base + trips * length : length]

    def earliest_trip(self, pattern: int, position: int, time: int) -> int:
        """Index of the first trip of ``pattern`` departing ``position`` at or
        after ``time``; ``-1`` if the service day is over for that pattern.

        Both branches rely on the non-overtaking invariant: departures down a
        position are ascending in trip order, so the first hit is the earliest.
        See :data:`LINEAR_SCAN_TRIPS` for why the short case does not binary
        search.
        """
        length = self._length[pattern]
        trips = self._trips[pattern]
        base = self._base[pattern] + position

        if trips <= LINEAR_SCAN_TRIPS:
            departures = self._dep
            index = base
            for trip in range(trips):
                if departures[index] >= time:
                    return trip
                index += length
            return -1

        column = self.departures[base : base + trips * length : length]
        hit = int(np.searchsorted(column, time, side="left"))
        return hit if hit < trips else -1

    def arrival_at(self, pattern: int, trip: int, position: int) -> int:
        return self._arr[self._base[pattern] + trip * self._length[pattern] + position]

    def departure_at(self, pattern: int, trip: int, position: int) -> int:
        return self._dep[self._base[pattern] + trip * self._length[pattern] + position]

    def trip_id_of(self, pattern: int, trip: int) -> str:
        return self.trip_ids[int(self.pattern_trip_offset[pattern]) + trip]

    def patterns_at(self, stop: int) -> Iterator[tuple[int, int]]:
        """Yield ``(pattern, position)`` for every pattern serving ``stop``."""
        lo = int(self.stop_pattern_offset[stop])
        hi = int(self.stop_pattern_offset[stop + 1])
        for k in range(lo, hi):
            yield int(self.stop_pattern_pattern[k]), int(self.stop_pattern_position[k])

    def footpaths_from(self, stop: int) -> Iterator[tuple[int, int]]:
        """Yield ``(target_stop, walk_seconds)``, excluding ``stop`` itself."""
        lo = int(self.footpath_offset[stop])
        hi = int(self.footpath_offset[stop + 1])
        for k in range(lo, hi):
            yield int(self.footpath_target[k]), int(self.footpath_time[k])

    def summary(self) -> str:
        return (
            f"{self.n_stops} stops, {self.n_patterns} patterns, "
            f"{self.n_trips} trips, {self.n_stop_times} stop-times, "
            f"{int(self.footpath_target.size)} footpaths"
        )


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------
def build_timetable(
    stops: Sequence[Stop],
    patterns: Sequence[RawPattern],
    footpaths: Iterable[tuple[str, str, int]] = (),
    *,
    max_footpath_seconds: int = MAX_FOOTPATH_SECONDS,
) -> Timetable:
    """Flatten loader output into the CSR arrays above.

    Stops keep the order given, so the caller controls stop indices (the
    loader sorts by ``stop_id`` to make indices reproducible across runs).
    """
    stop_tuple = tuple(stops)
    stop_index = {s.stop_id: i for i, s in enumerate(stop_tuple)}
    if len(stop_index) != len(stop_tuple):
        raise ValueError("duplicate stop_id in stops")
    n_stops = len(stop_tuple)

    coords = np.array([[s.lat, s.lon] for s in stop_tuple], dtype=np.float64) if n_stops else np.zeros((0, 2))

    n_patterns = len(patterns)
    pattern_stop_offset = np.zeros(n_patterns + 1, dtype=np.int32)
    pattern_trip_offset = np.zeros(n_patterns + 1, dtype=np.int32)
    stop_time_offset = np.zeros(n_patterns + 1, dtype=np.int64)

    for p, raw in enumerate(patterns):
        pattern_stop_offset[p + 1] = pattern_stop_offset[p] + len(raw.stops)
        pattern_trip_offset[p + 1] = pattern_trip_offset[p] + len(raw.trip_ids)
        stop_time_offset[p + 1] = stop_time_offset[p] + len(raw.stops) * len(raw.trip_ids)

    pattern_stops = np.zeros(int(pattern_stop_offset[-1]), dtype=np.int32)
    arrivals = np.zeros(int(stop_time_offset[-1]), dtype=np.int32)
    departures = np.zeros(int(stop_time_offset[-1]), dtype=np.int32)
    route_ids: list[str] = []
    trip_ids: list[str] = []

    for p, raw in enumerate(patterns):
        route_ids.append(raw.route_id)
        length = len(raw.stops)
        try:
            pattern_stops[pattern_stop_offset[p] : pattern_stop_offset[p + 1]] = [
                stop_index[s] for s in raw.stops
            ]
        except KeyError as exc:
            raise ValueError(f"pattern {raw.route_id} references unknown stop {exc}") from exc

        order = _sorted_trip_order(raw)
        trip_ids.extend(raw.trip_ids[t] for t in order)

        base = int(stop_time_offset[p])
        for slot, t in enumerate(order):
            lo = base + slot * length
            arrivals[lo : lo + length] = raw.arrivals[t]
            departures[lo : lo + length] = raw.departures[t]

        _check_monotone_within_trips(raw, arrivals, departures, base, length, len(order))
        _check_non_overtaking(raw, departures, base, length, len(order))

    stop_pattern_offset, stop_pattern_pattern, stop_pattern_position = _build_stop_index(
        n_stops, n_patterns, pattern_stop_offset, pattern_stops
    )

    edges = [
        (stop_index[a], stop_index[b], int(secs))
        for a, b, secs in footpaths
        if a in stop_index and b in stop_index and a != b
    ]
    footpath_offset, footpath_target, footpath_time = close_footpaths(
        n_stops, edges, max_footpath_seconds
    )

    return Timetable(
        stops=stop_tuple,
        stop_index=stop_index,
        coords=coords,
        pattern_route_id=tuple(route_ids),
        pattern_stop_offset=pattern_stop_offset,
        pattern_stops=pattern_stops,
        pattern_trip_offset=pattern_trip_offset,
        trip_ids=tuple(trip_ids),
        stop_time_offset=stop_time_offset,
        arrivals=arrivals,
        departures=departures,
        stop_pattern_offset=stop_pattern_offset,
        stop_pattern_pattern=stop_pattern_pattern,
        stop_pattern_position=stop_pattern_position,
        footpath_offset=footpath_offset,
        footpath_target=footpath_target,
        footpath_time=footpath_time,
    )


def _sorted_trip_order(raw: RawPattern) -> list[int]:
    """Trip slots ordered by departure at the first stop (ties by trip_id)."""
    return sorted(
        range(len(raw.trip_ids)),
        key=lambda t: (raw.departures[t][0], raw.trip_ids[t]),
    )


def _check_monotone_within_trips(
    raw: RawPattern,
    arrivals: np.ndarray,
    departures: np.ndarray,
    base: int,
    length: int,
    trips: int,
) -> None:
    """A trip may not arrive at a stop before it left the previous one."""
    for slot in range(trips):
        lo = base + slot * length
        arr = arrivals[lo : lo + length]
        dep = departures[lo : lo + length]
        if np.any(dep < arr):
            raise ValueError(f"pattern {raw.route_id}: departure before arrival")
        if np.any(arr[1:] < dep[:-1]):
            raise ValueError(f"pattern {raw.route_id}: stop times go backwards along a trip")


def _check_non_overtaking(
    raw: RawPattern,
    departures: np.ndarray,
    base: int,
    length: int,
    trips: int,
) -> None:
    """Verify the ordering chosen at the first stop holds at every position."""
    if trips < 2:
        return
    block = departures[base : base + trips * length].reshape(trips, length)
    bad = np.flatnonzero(np.any(np.diff(block, axis=0) < 0, axis=0))
    if bad.size:
        raise OvertakingError(
            f"pattern {raw.route_id}: trips overtake at position(s) "
            f"{bad.tolist()} -- split the pattern or the binary search for the "
            "earliest boardable trip is invalid"
        )


def _build_stop_index(
    n_stops: int,
    n_patterns: int,
    pattern_stop_offset: np.ndarray,
    pattern_stops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert pattern -> stops into stop -> (pattern, position), in CSR.

    Built by counting sort rather than by appending to per-stop lists: two
    passes over the same flat array, one allocation of the exact final size.
    """
    counts = np.zeros(n_stops + 1, dtype=np.int32)
    np.add.at(counts, pattern_stops.astype(np.int64) + 1, 1)
    offset = np.cumsum(counts, dtype=np.int32)

    total = int(offset[-1])
    out_pattern = np.zeros(total, dtype=np.int32)
    out_position = np.zeros(total, dtype=np.int32)
    cursor = offset[:-1].copy()

    for p in range(n_patterns):
        lo = int(pattern_stop_offset[p])
        hi = int(pattern_stop_offset[p + 1])
        for position, stop in enumerate(pattern_stops[lo:hi]):
            slot = int(cursor[stop])
            out_pattern[slot] = p
            out_position[slot] = position
            cursor[stop] = slot + 1

    return offset, out_pattern, out_position


def close_footpaths(
    n_stops: int,
    edges: Sequence[tuple[int, int, int]],
    max_seconds: int = MAX_FOOTPATH_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transitive closure of the walking graph, as CSR arrays.

    RAPTOR relaxes footpaths exactly once per round, so the walking graph has
    to be closed up front: if A->B and B->C are both walkable, A->C must be an
    edge or the router can miss a journey that walks twice.

    Done with one bounded Dijkstra per source, not Floyd-Warshall. Closure is
    an all-pairs shortest-path problem, but the ``max_seconds`` bound keeps
    each search inside a handful of nearby stops, so the cost is
    ``O(S * k log k)`` with ``k`` tiny, instead of ``O(S^3)`` -- at 10k stops
    Floyd-Warshall is a trillion operations and 400 MB of distance matrix.

    Self-links are left out; the router treats the origin's own stop
    explicitly. Each source's list is sorted by walking time so a caller may
    stop early once the walk exceeds its budget.
    """
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_stops)]
    for u, v, w in edges:
        if w <= max_seconds:
            adjacency[u].append((v, int(w)))

    offset = np.zeros(n_stops + 1, dtype=np.int32)
    targets: list[int] = []
    times: list[int] = []

    for src in range(n_stops):
        best: dict[int, int] = {src: 0}
        queue: list[tuple[int, int]] = [(0, src)]
        settled: list[tuple[int, int]] = []
        while queue:
            dist, node = heapq.heappop(queue)
            if dist > best.get(node, dist):
                continue  # stale heap entry
            if node != src:
                settled.append((dist, node))
            for nxt, weight in adjacency[node]:
                cand = dist + weight
                if cand <= max_seconds and cand < best.get(nxt, max_seconds + 1):
                    best[nxt] = cand
                    heapq.heappush(queue, (cand, nxt))
        settled.sort()
        for dist, node in settled:
            targets.append(node)
            times.append(dist)
        offset[src + 1] = len(targets)

    return (
        offset,
        np.array(targets, dtype=np.int32),
        np.array(times, dtype=np.int32),
    )


def timetable_from_feed(feed, *, max_footpath_seconds: int = MAX_FOOTPATH_SECONDS) -> Timetable:
    """Build directly from a :class:`transitlab.gtfs.synthetic.Feed`.

    A shortcut for tests and benchmarks that want the in-memory network
    without the zip/DuckDB round trip. The loader path is what production
    uses, and :mod:`tests` asserts the two agree.
    """
    stops = [
        Stop(stop_id, name, lat, lon)
        for stop_id, (name, lat, lon) in sorted(feed.stops.items())
    ]
    raw_patterns: list[RawPattern] = []
    for p_idx, pattern in enumerate(feed.patterns):
        trip_ids: list[str] = []
        arrivals: list[list[int]] = []
        departures: list[list[int]] = []
        for t_idx, start in enumerate(pattern.departures):
            trip_ids.append(f"{pattern.route_id}_p{p_idx}_t{t_idx}")
            arr: list[int] = []
            dep: list[int] = []
            clock = start
            last = len(pattern.stops) - 1
            for seq in range(len(pattern.stops)):
                if seq == 0:
                    arr.append(clock)
                    dep.append(clock)
                else:
                    clock += pattern.hops[seq - 1]
                    arr.append(clock)
                    clock = clock if seq == last else clock + pattern.dwell
                    dep.append(clock)
            arrivals.append(arr)
            departures.append(dep)
        raw_patterns.append(
            RawPattern(
                route_id=pattern.route_id,
                stops=list(pattern.stops),
                trip_ids=trip_ids,
                arrivals=arrivals,
                departures=departures,
            )
        )
    return build_timetable(
        stops,
        raw_patterns,
        [(a, b, secs) for a, b, secs in feed.transfers],
        max_footpath_seconds=max_footpath_seconds,
    )
