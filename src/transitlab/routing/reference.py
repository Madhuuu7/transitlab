"""A time-expanded Dijkstra, used as the correctness oracle and the baseline.

This exists to be *wrong differently*. RAPTOR's answers are only convincing if
something that shares none of its machinery agrees with them, so this module
reimplements earliest-arrival routing as a shortest path over an explicit
time-expanded graph: no rounds, no patterns, no binary search over departure
columns, no reliance on the non-overtaking invariant.

The graph
---------
Every stop-time becomes two nodes, an **arrival** event and a **departure**
event, each with a fixed clock time. Edges only ever point forward in time:

* ``departure(trip, i) -> arrival(trip, i + 1)`` -- board here and ride a hop.
* ``arrival(trip, i) -> arrival(trip, i + 1)`` -- stay on the vehicle through
  a stop. This deliberately bypasses the departure event: letting an on-board
  passenger step onto the stop's waiting chain would let them change vehicles
  without paying ``change_time``, which made this graph disagree with RAPTOR
  by exactly one change time until it was found.
* ``arrival(trip, i) -> next departure at that stop at or after
  arrival + change_time`` -- alight and transfer.
* ``departure -> next departure at the same stop`` -- wait. Chaining
  consecutive departures is what keeps the graph sparse; connecting every
  arrival to every later departure at a stop would be quadratic.
* ``arrival(trip, i) -> next departure at a walkable stop at or after
  arrival + walk_time`` -- footpath.

Because each event node carries its own clock time and no edge goes backwards,
the shortest-path label of a node is simply that node's time. Dijkstra over
this graph therefore degenerates into a sweep of reachable events in time
order -- which is the honest way to describe it, and is still textbook
Dijkstra with a non-negative weight ``time[v] - time[u]`` on every edge.

What this does and does not prove
---------------------------------
It validates the router. It does **not** independently validate the footpath
transitive closure, because it walks the same closed footpath set out of the
timetable; the closure is checked separately against a Floyd-Warshall
all-pairs computation on small graphs, where cubic cost is affordable.

As a baseline it is also the thing RAPTOR is benchmarked against. The size
numbers in that comparison are the point: this graph holds two nodes per
stop-time and an edge per possible transfer, while RAPTOR runs over the same
feed with no graph at all.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from ..gtfs.times import NEVER
from ..gtfs.timetable import Timetable


@dataclass
class GraphSize:
    nodes: int
    edges: int

    def __str__(self) -> str:
        return f"{self.nodes} nodes, {self.edges} edges"


class TimeExpandedGraph:
    """Explicit event graph over a :class:`Timetable`.

    Built once per ``(change_time, max_walk_seconds)`` pair and reused across
    queries, the same way the RAPTOR timetable is.
    """

    def __init__(
        self,
        timetable: Timetable,
        *,
        change_time: int = 0,
        max_walk_seconds: int | None = None,
    ) -> None:
        self.timetable = timetable
        self.change_time = int(change_time)
        self.max_walk_seconds = max_walk_seconds

        tt = timetable
        n_events = 2 * tt.n_stop_times
        self.node_time = np.full(n_events, NEVER, dtype=np.int64)
        self.node_stop = np.full(n_events, -1, dtype=np.int32)
        self.is_arrival = np.zeros(n_events, dtype=bool)
        self.adjacency: list[list[int]] = [[] for _ in range(n_events)]

        # Departure events per stop, collected then sorted by time: the
        # waiting chain and every transfer edge snap into this list.
        departures_at: list[list[tuple[int, int]]] = [[] for _ in range(tt.n_stops)]
        arrival_nodes: list[list[int]] = [[] for _ in range(tt.n_stops)]

        for pattern in range(tt.n_patterns):
            stops = tt.stops_of(pattern)
            length = int(stops.size)
            base = int(tt.stop_time_offset[pattern])
            for trip in range(tt.trip_count(pattern)):
                for position in range(length):
                    index = base + trip * length + position
                    arrival_node = 2 * index
                    departure_node = 2 * index + 1
                    stop = int(stops[position])

                    self.node_time[arrival_node] = int(tt.arrivals[index])
                    self.node_time[departure_node] = int(tt.departures[index])
                    self.node_stop[arrival_node] = stop
                    self.node_stop[departure_node] = stop
                    self.is_arrival[arrival_node] = True
                    arrival_nodes[stop].append(arrival_node)

                    if position + 1 < length:
                        # ride on through (no change time, no waiting chain),
                        # or ride after boarding at this stop
                        self.adjacency[arrival_node].append(2 * (index + 1))
                        self.adjacency[departure_node].append(2 * (index + 1))
                        departures_at[stop].append(
                            (int(tt.departures[index]), departure_node)
                        )

        self.departure_times: list[np.ndarray] = []
        self.departure_nodes: list[np.ndarray] = []
        for stop in range(tt.n_stops):
            events = sorted(departures_at[stop])
            times = np.array([t for t, _ in events], dtype=np.int64)
            nodes = np.array([n for _, n in events], dtype=np.int64)
            self.departure_times.append(times)
            self.departure_nodes.append(nodes)
            # waiting: skip from one departure at this stop to the next
            for k in range(len(events) - 1):
                self.adjacency[int(nodes[k])].append(int(nodes[k + 1]))

        self.arrival_nodes = [np.array(v, dtype=np.int64) for v in arrival_nodes]

        # Transfer and footpath edges out of every arrival event.
        for stop in range(tt.n_stops):
            walks = [
                (target, seconds)
                for target, seconds in tt.footpaths_from(stop)
                if max_walk_seconds is None or seconds <= max_walk_seconds
            ]
            for arrival_node in self.arrival_nodes[stop]:
                arrival_time = int(self.node_time[arrival_node])
                board = self._next_departure(stop, arrival_time + self.change_time)
                if board >= 0:
                    self.adjacency[int(arrival_node)].append(board)
                for target, seconds in walks:
                    board = self._next_departure(target, arrival_time + seconds)
                    if board >= 0:
                        self.adjacency[int(arrival_node)].append(board)

        self.size = GraphSize(
            nodes=n_events, edges=sum(len(a) for a in self.adjacency)
        )

    def _next_departure(self, stop: int, time: int) -> int:
        """Node id of the first departure at ``stop`` at or after ``time``."""
        times = self.departure_times[stop]
        if times.size == 0:
            return -1
        index = int(np.searchsorted(times, time, side="left"))
        if index >= times.size:
            return -1
        return int(self.departure_nodes[stop][index])

    # ------------------------------------------------------------------
    def earliest_arrivals(
        self,
        origin: int | str,
        departure_time: int,
        destination: int | str | None = None,
    ) -> np.ndarray:
        """Earliest arrival time at every stop, as an ``int64`` array.

        ``destination`` only stops the sweep early; the returned array is then
        valid at that stop and a lower bound elsewhere.
        """
        tt = self.timetable
        source = tt.stop_index[origin] if isinstance(origin, str) else int(origin)
        target = (
            tt.stop_index[destination]
            if isinstance(destination, str)
            else (int(destination) if destination is not None else None)
        )

        best = np.full(tt.n_stops, NEVER, dtype=np.int64)
        best[source] = departure_time
        visited = np.zeros(self.size.nodes, dtype=bool)
        queue: list[tuple[int, int]] = []

        def seed(stop: int, time: int) -> None:
            node = self._next_departure(stop, time)
            if node >= 0:
                heapq.heappush(queue, (int(self.node_time[node]), node))

        seed(source, departure_time)
        for neighbour, seconds in tt.footpaths_from(source):
            if self.max_walk_seconds is not None and seconds > self.max_walk_seconds:
                continue
            arrival = departure_time + seconds
            if arrival < best[neighbour]:
                best[neighbour] = arrival
            seed(neighbour, arrival)

        while queue:
            time, node = heapq.heappop(queue)
            if visited[node]:
                continue
            visited[node] = True

            if target is not None and time > best[target]:
                break  # every remaining event is later than the answer

            if self.is_arrival[node]:
                stop = int(self.node_stop[node])
                if time < best[stop]:
                    best[stop] = time
                # Walking off the vehicle is an arrival in its own right, even
                # when no onward departure exists to snap to.
                for neighbour, seconds in tt.footpaths_from(stop):
                    if (
                        self.max_walk_seconds is not None
                        and seconds > self.max_walk_seconds
                    ):
                        continue
                    if time + seconds < best[neighbour]:
                        best[neighbour] = time + seconds

            for nxt in self.adjacency[node]:
                if not visited[nxt]:
                    heapq.heappush(queue, (int(self.node_time[nxt]), nxt))

        return best


def earliest_arrivals(
    timetable: Timetable,
    origin: int | str,
    departure_time: int,
    *,
    change_time: int = 0,
    max_walk_seconds: int | None = None,
) -> np.ndarray:
    """One-shot reference query. Rebuilds the graph, so it is for tests."""
    graph = TimeExpandedGraph(
        timetable, change_time=change_time, max_walk_seconds=max_walk_seconds
    )
    return graph.earliest_arrivals(origin, departure_time)


def floyd_warshall_closure(
    n_stops: int,
    edges: list[tuple[int, int, int]],
    max_seconds: int,
) -> dict[tuple[int, int], int]:
    """All-pairs walking times by Floyd-Warshall, for testing the closure.

    Cubic, so only usable on toy graphs -- which is exactly why the real
    closure uses bounded Dijkstra instead. Returns only pairs within the
    bound, excluding self-pairs, to match
    :func:`transitlab.gtfs.timetable.close_footpaths`.
    """
    infinity = float("inf")
    dist = [[infinity] * n_stops for _ in range(n_stops)]
    for i in range(n_stops):
        dist[i][i] = 0
    for u, v, w in edges:
        if w <= max_seconds and w < dist[u][v]:
            dist[u][v] = w

    for k in range(n_stops):
        row_k = dist[k]
        for i in range(n_stops):
            via = dist[i][k]
            if via == infinity:
                continue
            row_i = dist[i]
            for j in range(n_stops):
                candidate = via + row_k[j]
                if candidate < row_i[j]:
                    row_i[j] = candidate

    return {
        (i, j): int(dist[i][j])
        for i in range(n_stops)
        for j in range(n_stops)
        if i != j and dist[i][j] <= max_seconds
    }
