"""RAPTOR -- Round-bAsed Public Transit Optimized Router.

Implements the earliest-arrival variant from Delling, Pajor and Werneck,
*Round-Based Public Transit Routing* (2012), over the flat arrays in
:mod:`transitlab.gtfs.timetable`.

The idea
--------
Dijkstra on a time-expanded graph treats "wait at a stop" and "ride one hop"
as edges and spends most of its time in a priority queue over millions of
nodes. RAPTOR throws the queue away. It works in **rounds**: after round ``k``,
``tau[k][stop]`` is the earliest time ``stop`` can be reached using at most
``k`` vehicle trips. Each round does two passes:

1. **Scan patterns.** Every pattern touching a stop improved in the previous
   round is traversed once, in stop order, carrying "the earliest trip I could
   be on". Boarding earlier is only ever decided at a stop and alighting is
   just a comparison, so one linear sweep per pattern per round suffices.
2. **Relax footpaths.** Walking links are applied once per round, which is
   correct only because the walking graph was transitively closed at build
   time.

The round index *is* the transfer count, which makes the derived queries cheap:
a one-to-all run answers isochrones, and the per-round table is the Pareto
front over (arrival, transfers).

Two labels per stop, not one
----------------------------
Textbook RAPTOR keeps a single label per stop and prunes any arrival that does
not improve it. That is sound only if a stop reached on foot can walk onward
as freely as a stop reached by vehicle. Our footpath closure is **bounded**
(:data:`transitlab.gtfs.timetable.MAX_FOOTPATH_SECONDS`), so it is not: a walk
of A->B->C may be missing from the closure because the chain exceeds the
bound, while A->B is present.

With one label that loses journeys. Walking from the origin to B at 04:06
suppresses a bus that reaches B at 06:38, and the walk B->C that only the bus
arrival is entitled to make is never relaxed -- so C looks unreachable when it
is not. Differential testing against
:mod:`transitlab.routing.reference` caught exactly that case.

The fix is to label each stop twice: ``tau_v``, the best arrival whose last
leg is a vehicle, and ``tau_f``, the best arrival whose last leg is a walk.
Footpaths are relaxed only out of ``tau_v`` (and out of the origin, which is
where the passenger already stands), while boarding may use either -- with the
change time charged only after a ride, since a walking transfer has already
paid its own time. ``tau`` is the minimum of the two and is what queries read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..gtfs.times import NEVER, format_gtfs_time
from ..gtfs.timetable import Timetable


@dataclass(frozen=True)
class TripLeg:
    """A ride on one vehicle, boarding and alighting at given stops."""

    route_id: str
    trip_id: str
    from_stop: str
    to_stop: str
    departure: int
    arrival: int

    @property
    def duration(self) -> int:
        return self.arrival - self.departure

    def __str__(self) -> str:
        return (
            f"{format_gtfs_time(self.departure)} {self.from_stop} "
            f"--{self.route_id}--> {self.to_stop} {format_gtfs_time(self.arrival)}"
        )


@dataclass(frozen=True)
class WalkLeg:
    from_stop: str
    to_stop: str
    seconds: int

    def __str__(self) -> str:
        return f"walk {self.from_stop} -> {self.to_stop} ({self.seconds // 60} min)"


@dataclass(frozen=True)
class Journey:
    """A reconstructed journey: alternating walks and rides."""

    departure: int
    arrival: int
    legs: tuple[TripLeg | WalkLeg, ...]

    @property
    def duration(self) -> int:
        return self.arrival - self.departure

    @property
    def n_transfers(self) -> int:
        rides = sum(1 for leg in self.legs if isinstance(leg, TripLeg))
        return max(rides - 1, 0)

    @property
    def walk_seconds(self) -> int:
        return sum(leg.seconds for leg in self.legs if isinstance(leg, WalkLeg))

    def __str__(self) -> str:
        head = (
            f"{format_gtfs_time(self.departure)} -> {format_gtfs_time(self.arrival)} "
            f"({self.duration // 60} min, {self.n_transfers} transfers)"
        )
        return "\n".join([head, *(f"  {leg}" for leg in self.legs)])


class RaptorResult:
    """One-to-all result of a single RAPTOR run.

    Holds the per-round arrival table and the parent pointers needed to
    reconstruct a journey to *any* stop, because a run that answers one
    destination has already answered them all -- which is what makes
    isochrones nearly free.
    """

    def __init__(
        self,
        timetable: Timetable,
        origin: int,
        departure_time: int,
        tau: np.ndarray,
        tau_vehicle: np.ndarray,
        tau_foot: np.ndarray,
        best: np.ndarray,
        parents: dict[str, np.ndarray],
        rounds_used: int,
    ) -> None:
        self.timetable = timetable
        self.origin = origin
        self.departure_time = departure_time
        self.by_round = tau
        self.tau_vehicle = tau_vehicle
        self.tau_foot = tau_foot
        self.best = best
        self._parents = parents
        self.rounds_used = rounds_used

    # -- queries --------------------------------------------------------
    def arrival(self, stop: int | str) -> int:
        """Earliest arrival at ``stop``, or :data:`NEVER` if unreachable."""
        return int(self.best[self._resolve(stop)])

    def reachable(self, within_seconds: int) -> np.ndarray:
        """Stop indices reachable within a time budget -- the isochrone set."""
        limit = self.departure_time + within_seconds
        return np.flatnonzero(self.best <= limit)

    def journey(self, stop: int | str) -> Journey | None:
        """Reconstruct the earliest-arrival journey to ``stop``."""
        target = self._resolve(stop)
        arrival = int(self.best[target])
        if arrival >= NEVER:
            return None
        round_index = int(np.argmax(self.by_round[:, target] == arrival))
        return self._build(target, round_index, arrival)

    def journeys_by_transfers(self, stop: int | str) -> list[Journey]:
        """One journey per transfer count, dropping those no faster.

        The per-round table already holds "fastest with at most k trips", so
        keeping a round only when it beats every earlier round yields the
        Pareto front over (arrival time, transfers) at no extra search cost.
        """
        target = self._resolve(stop)
        out: list[Journey] = []
        incumbent = NEVER
        for round_index in range(self.rounds_used + 1):
            arrival = int(self.by_round[round_index, target])
            if arrival >= incumbent or arrival >= NEVER:
                continue
            incumbent = arrival
            journey = self._build(target, round_index, arrival)
            if journey is not None and journey.legs:
                out.append(journey)
        return out

    # -- internals ------------------------------------------------------
    def _resolve(self, stop: int | str) -> int:
        return self.timetable.stop_index[stop] if isinstance(stop, str) else int(stop)

    def _build(self, target: int, round_index: int, arrival: int) -> Journey | None:
        legs = list(self._walk_back(target, round_index))
        legs.reverse()
        return Journey(
            departure=self.departure_time, arrival=arrival, legs=tuple(legs)
        )

    def _walk_back(self, target: int, round_index: int) -> Iterator[TripLeg | WalkLeg]:
        """Yield the legs of a journey backwards from ``target``.

        ``by_foot`` tracks which of the two labels is being followed. A walk
        leg must have come out of a vehicle arrival (or the origin), so the
        step after one always follows the vehicle label; a ride drops the round
        index by one, which bounds the loop.
        """
        tt = self.timetable
        parents = self._parents
        stop = target
        k = round_index
        by_foot = int(self.tau_foot[k, stop]) == int(self.by_round[k, stop])
        guard = 0

        while True:
            if stop == self.origin and int(parents["pattern"][k, stop]) < 0:
                return
            guard += 1
            if guard > 2 * (self.rounds_used + 1) * (tt.n_stops + 1):
                raise RuntimeError("journey reconstruction failed to terminate")

            if by_foot:
                source = int(parents["walk_from"][k, stop])
                seconds = int(parents["walk_secs"][k, stop])
                if source < 0:
                    raise RuntimeError(
                        f"no walk parent for stop {stop} in round {k}"
                    )
                yield WalkLeg(
                    from_stop=tt.stops[source].stop_id,
                    to_stop=tt.stops[stop].stop_id,
                    seconds=seconds,
                )
                stop = source
                by_foot = False
                continue

            pattern = int(parents["pattern"][k, stop])
            if pattern < 0:
                raise RuntimeError(
                    f"no trip parent for stop {stop} in round {k}; labels and "
                    "parent pointers disagree"
                )
            trip = int(parents["trip"][k, stop])
            board = int(parents["board"][k, stop])
            alight = int(parents["alight"][k, stop])
            source = int(tt.stops_of(pattern)[board])
            departure = tt.departure_at(pattern, trip, board)
            yield TripLeg(
                route_id=tt.pattern_route_id[pattern],
                trip_id=tt.trip_id_of(pattern, trip),
                from_stop=tt.stops[source].stop_id,
                to_stop=tt.stops[stop].stop_id,
                departure=departure,
                arrival=tt.arrival_at(pattern, trip, alight),
            )
            stop = source
            k -= 1
            if stop == self.origin and int(parents["pattern"][k, stop]) < 0:
                return
            # Which label did this boarding use? Either is a valid predecessor
            # as long as it was early enough; prefer the walk, since a foot
            # label is what a transfer over a footpath looks like.
            foot = int(self.tau_foot[k, stop])
            by_foot = foot < NEVER and foot <= departure

    # -- for tests ------------------------------------------------------
    def label_tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.by_round, self.tau_vehicle, self.tau_foot


class Raptor:
    """Earliest-arrival RAPTOR over a prepared :class:`Timetable`."""

    def __init__(
        self,
        timetable: Timetable,
        *,
        min_change_time: int = 0,
        max_rounds: int = 8,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.timetable = timetable
        self.min_change_time = int(min_change_time)
        self.max_rounds = int(max_rounds)

    def run(
        self,
        origin: int | str,
        departure_time: int,
        destination: int | str | None = None,
        *,
        max_walk_seconds: int | None = None,
    ) -> RaptorResult:
        """Earliest arrival at every stop, leaving ``origin`` at or after
        ``departure_time``.

        Passing ``destination`` enables target pruning: a label that cannot
        beat the best known arrival at the destination is never written, which
        removes most of the work on a long one-to-one query without changing
        the answer there. Arrivals at other stops are then only lower bounds,
        so a journey to a different stop needs a run without it.
        """
        tt = self.timetable
        source = tt.stop_index[origin] if isinstance(origin, str) else int(origin)
        target = (
            tt.stop_index[destination]
            if isinstance(destination, str)
            else (int(destination) if destination is not None else None)
        )
        change = self.min_change_time
        rounds = self.max_rounds
        n_stops = tt.n_stops

        tau = np.full((rounds + 1, n_stops), NEVER, dtype=np.int32)
        tau_vehicle = np.full((rounds + 1, n_stops), NEVER, dtype=np.int32)
        tau_foot = np.full((rounds + 1, n_stops), NEVER, dtype=np.int32)
        best = np.full(n_stops, NEVER, dtype=np.int32)
        best_vehicle = np.full(n_stops, NEVER, dtype=np.int32)
        best_foot = np.full(n_stops, NEVER, dtype=np.int32)
        parents = {
            "pattern": np.full((rounds + 1, n_stops), -1, dtype=np.int32),
            "trip": np.full((rounds + 1, n_stops), -1, dtype=np.int32),
            "board": np.full((rounds + 1, n_stops), -1, dtype=np.int32),
            "alight": np.full((rounds + 1, n_stops), -1, dtype=np.int32),
            "walk_from": np.full((rounds + 1, n_stops), -1, dtype=np.int32),
            "walk_secs": np.zeros((rounds + 1, n_stops), dtype=np.int32),
        }

        # The passenger starts standing at the origin, so it counts as
        # walk-capable (tau_vehicle) and needs no change time to board
        # (tau_foot). Both, in other words.
        tau[0, source] = departure_time
        tau_vehicle[0, source] = departure_time
        tau_foot[0, source] = departure_time
        best[source] = best_vehicle[source] = best_foot[source] = departure_time
        improved: set[int] = {source}

        for neighbour, seconds in tt.footpaths_from(source):
            if max_walk_seconds is not None and seconds > max_walk_seconds:
                continue
            candidate = departure_time + seconds
            if candidate < tau_foot[0, neighbour]:
                tau_foot[0, neighbour] = candidate
                best_foot[neighbour] = candidate
                if candidate < tau[0, neighbour]:
                    tau[0, neighbour] = candidate
                if candidate < best[neighbour]:
                    best[neighbour] = candidate
                parents["walk_from"][0, neighbour] = source
                parents["walk_secs"][0, neighbour] = seconds
                improved.add(neighbour)

        rounds_used = 0
        for k in range(1, rounds + 1):
            # Carry round k-1 forward *with its parents*, so every label in
            # row k has a reconstructable predecessor and row k means "at most
            # k trips" directly.
            tau[k] = tau[k - 1]
            tau_vehicle[k] = tau_vehicle[k - 1]
            tau_foot[k] = tau_foot[k - 1]
            for array in parents.values():
                array[k] = array[k - 1]

            # Stage 1: each pattern through an improved stop, entered at the
            # earliest such position. Entering earlier cannot help: nothing
            # before it improved last round.
            queue: dict[int, int] = {}
            for stop in improved:
                for pattern, position in tt.patterns_at(stop):
                    current = queue.get(pattern)
                    if current is None or position < current:
                        queue[pattern] = position
            if not queue:
                break

            improved = set()
            rode_in: list[int] = []
            bound = int(best[target]) if target is not None else NEVER

            for pattern, start in queue.items():
                stops = tt.stops_of(pattern)
                length = int(stops.size)
                trip = -1
                board = -1

                for position in range(start, length):
                    stop = int(stops[position])

                    if trip >= 0:
                        arrival = tt.arrival_at(pattern, trip, position)
                        if arrival < min(int(best_vehicle[stop]), bound):
                            tau_vehicle[k, stop] = arrival
                            best_vehicle[stop] = arrival
                            if arrival < tau[k, stop]:
                                tau[k, stop] = arrival
                            if arrival < best[stop]:
                                best[stop] = arrival
                            parents["pattern"][k, stop] = pattern
                            parents["trip"][k, stop] = trip
                            parents["board"][k, stop] = board
                            parents["alight"][k, stop] = position
                            improved.add(stop)
                            rode_in.append(stop)
                            if target is not None and stop == target:
                                bound = min(bound, arrival)

                    # Boarding uses the *previous* round's labels: arriving
                    # here in this round means arriving by this very scan, and
                    # boarding again would not be a transfer. The change time
                    # is charged only after a ride -- a walking transfer has
                    # already paid its own time.
                    ready = int(tau_foot[k - 1, stop])
                    riding = int(tau_vehicle[k - 1, stop])
                    if riding < NEVER and riding + change < ready:
                        ready = riding + change
                    if ready >= NEVER:
                        continue
                    if trip < 0 or ready <= tt.departure_at(pattern, trip, position):
                        candidate = tt.earliest_trip(pattern, position, ready)
                        if candidate >= 0 and (trip < 0 or candidate < trip):
                            trip = candidate
                            board = position

            # Stage 2: one footpath pass, out of the stops a vehicle just
            # reached. Transitive closure means a single pass cannot miss a
            # walk-then-walk connection that is within the bound; a chain that
            # exceeds the bound is deliberately not a connection.
            for stop in rode_in:
                arrival = int(tau_vehicle[k, stop])
                for neighbour, seconds in tt.footpaths_from(stop):
                    if max_walk_seconds is not None and seconds > max_walk_seconds:
                        continue
                    candidate = arrival + seconds
                    if candidate < min(int(best_foot[neighbour]), bound):
                        tau_foot[k, neighbour] = candidate
                        best_foot[neighbour] = candidate
                        if candidate < tau[k, neighbour]:
                            tau[k, neighbour] = candidate
                        if candidate < best[neighbour]:
                            best[neighbour] = candidate
                        parents["walk_from"][k, neighbour] = stop
                        parents["walk_secs"][k, neighbour] = seconds
                        improved.add(neighbour)

            if not improved:
                break
            rounds_used = k

        return RaptorResult(
            timetable=tt,
            origin=source,
            departure_time=departure_time,
            tau=tau,
            tau_vehicle=tau_vehicle,
            tau_foot=tau_foot,
            best=best,
            parents=parents,
            rounds_used=rounds_used,
        )

    def journey(
        self,
        origin: int | str,
        destination: int | str,
        departure_time: int,
        **kwargs: object,
    ) -> Journey | None:
        """Convenience one-to-one query."""
        result = self.run(origin, departure_time, destination, **kwargs)  # type: ignore[arg-type]
        return result.journey(destination)
