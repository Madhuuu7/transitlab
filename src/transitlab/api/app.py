"""HTTP API over the routing engine.

The engine is loaded once at startup and shared: the timetable is immutable
NumPy arrays, every query allocates its own label table, so concurrent
requests never touch each other's state. Loading per request would cost the
whole feed ingest each time.

Endpoints are deliberately thin. All the interesting work -- pattern
extraction, the closure, the rounds -- happens below this layer, and the
response models exist so the frontend never has to parse GTFS times.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..gtfs.loader import FeedError, LoadStats, load_timetable
from ..gtfs.snapshot import SnapshotError
from ..gtfs.snapshot import load as load_snapshot
from ..gtfs.times import NEVER, format_gtfs_time, parse_gtfs_time
from ..gtfs.timetable import Timetable
from ..routing.raptor import Journey, Raptor, TripLeg

DEFAULT_FEED = Path("data/sample-feed.zip")
# The map UI is one static page with no build step, served by this app so the
# whole project deploys as a single service on a free tier.
WEB_ROOT = Path(__file__).resolve().parents[3] / "web"


# ----------------------------------------------------------------------
# response models
# ----------------------------------------------------------------------
class StopOut(BaseModel):
    stop_id: str
    name: str
    lat: float
    lon: float


class LegOut(BaseModel):
    kind: Literal["ride", "walk"]
    from_stop: str
    to_stop: str
    from_name: str
    to_name: str
    departure: str | None = None
    arrival: str | None = None
    route_id: str | None = None
    trip_id: str | None = None
    seconds: int


class JourneyOut(BaseModel):
    departure: str
    arrival: str
    duration_minutes: int
    transfers: int
    walk_minutes: int
    legs: list[LegOut]


class JourneyResponse(BaseModel):
    origin: StopOut
    destination: StopOut
    requested_departure: str
    fastest: JourneyOut | None
    alternatives: list[JourneyOut] = Field(
        default_factory=list,
        description="One journey per transfer count, each faster than the last",
    )


class ReachedStop(BaseModel):
    stop_id: str
    name: str
    lat: float
    lon: float
    arrival: str
    minutes: int


class IsochroneResponse(BaseModel):
    origin: StopOut
    departure: str
    budget_minutes: int
    reached: list[ReachedStop]
    unreachable: int


class FeedInfo(BaseModel):
    feed: str
    stops: int
    patterns: int
    trips: int
    stop_times: int
    footpaths: int


# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------
class Engine:
    """A loaded timetable plus the router configured over it."""

    def __init__(self, timetable: Timetable, feed_name: str, stats: LoadStats) -> None:
        self.timetable = timetable
        self.feed_name = feed_name
        self.stats = stats

    def router(self, change_time: int) -> Raptor:
        return Raptor(self.timetable, min_change_time=change_time)

    def resolve(self, stop_id: str, label: str) -> int:
        try:
            return self.timetable.stop_index[stop_id]
        except KeyError:
            raise HTTPException(404, f"unknown {label} stop: {stop_id}") from None

    def stop_out(self, index: int) -> StopOut:
        stop = self.timetable.stops[index]
        return StopOut(stop_id=stop.stop_id, name=stop.name, lat=stop.lat, lon=stop.lon)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Load the feed named by ``TRANSITLAB_FEED`` once per process.

    A ``.npz`` is a prebuilt snapshot and is loaded directly; anything else is
    a GTFS feed and is ingested here. Prefer the snapshot in a deployment: a
    city-sized feed costs seconds and over a gigabyte to ingest and 15 MB to
    hold, so paying the ingest on every boot sizes the instance for work it
    does once. See :mod:`transitlab.gtfs.snapshot`.

    ``TRANSITLAB_WALK_RADIUS_M`` adds walking transfers between stops within
    that many metres. It is needed on any feed that ships no ``transfers.txt``
    -- without it the router cannot cross a road, which on a feed that models
    each direction of a street as its own stop produces journeys with
    absurd numbers of changes. It applies to ingest only; a snapshot already
    has whatever walking graph it was built with.
    """
    feed = Path(os.environ.get("TRANSITLAB_FEED", str(DEFAULT_FEED)))
    date = os.environ.get("TRANSITLAB_DATE") or None
    radius = float(os.environ.get("TRANSITLAB_WALK_RADIUS_M") or 0) or None
    stats = LoadStats()

    if feed.suffix == ".npz":
        try:
            timetable = load_snapshot(feed)
        except (SnapshotError, OSError) as exc:  # pragma: no cover - startup path
            raise RuntimeError(f"could not load snapshot {feed}: {exc}") from exc
        stats.stops = timetable.n_stops
        stats.patterns = timetable.n_patterns
        stats.stop_times = timetable.n_stop_times
        return Engine(timetable, feed.name, stats)

    try:
        timetable = load_timetable(feed, date=date, walk_radius_m=radius, stats=stats)
    except FeedError as exc:  # pragma: no cover - startup failure path
        raise RuntimeError(f"could not load {feed}: {exc}") from exc
    return Engine(timetable, feed.name, stats)


EngineDep = Annotated[Engine, Depends(get_engine)]


# ----------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------
def _leg_out(engine: Engine, leg: object) -> LegOut:
    timetable = engine.timetable
    name = lambda stop_id: timetable.stops[timetable.stop_index[stop_id]].name  # noqa: E731
    if isinstance(leg, TripLeg):
        return LegOut(
            kind="ride",
            from_stop=leg.from_stop,
            to_stop=leg.to_stop,
            from_name=name(leg.from_stop),
            to_name=name(leg.to_stop),
            departure=format_gtfs_time(leg.departure),
            arrival=format_gtfs_time(leg.arrival),
            route_id=leg.route_id,
            trip_id=leg.trip_id,
            seconds=leg.duration,
        )
    return LegOut(
        kind="walk",
        from_stop=leg.from_stop,  # type: ignore[union-attr]
        to_stop=leg.to_stop,  # type: ignore[union-attr]
        from_name=name(leg.from_stop),  # type: ignore[union-attr]
        to_name=name(leg.to_stop),  # type: ignore[union-attr]
        seconds=leg.seconds,  # type: ignore[union-attr]
    )


def _journey_out(engine: Engine, journey: Journey) -> JourneyOut:
    return JourneyOut(
        departure=format_gtfs_time(journey.departure),
        arrival=format_gtfs_time(journey.arrival),
        duration_minutes=round(journey.duration / 60),
        transfers=journey.n_transfers,
        walk_minutes=round(journey.walk_seconds / 60),
        legs=[_leg_out(engine, leg) for leg in journey.legs],
    )


def _parse_time(value: str) -> int:
    """Accept HH:MM as well as HH:MM:SS, keeping hours past 24 legal."""
    text = value.strip()
    if text.count(":") == 1:
        text = f"{text}:00"
    try:
        return parse_gtfs_time(text)
    except ValueError as exc:
        raise HTTPException(422, f"bad time {value!r}: {exc}") from None


# ----------------------------------------------------------------------
# app
# ----------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="transitlab",
        version="0.1.0",
        description=(
            "Multi-criteria transit journey planning. The router is RAPTOR, "
            "implemented from scratch over flat NumPy arrays and verified "
            "against a time-expanded Dijkstra."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # a public read-only API over an open data feed
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/feed", response_model=FeedInfo)
    def feed(engine: EngineDep) -> FeedInfo:
        timetable = engine.timetable
        return FeedInfo(
            feed=engine.feed_name,
            stops=timetable.n_stops,
            patterns=timetable.n_patterns,
            trips=timetable.n_trips,
            stop_times=timetable.n_stop_times,
            footpaths=int(timetable.footpath_target.size),
        )

    @app.get("/stops", response_model=list[StopOut])
    def stops(
        engine: EngineDep,
        q: str | None = Query(None, description="case-insensitive name filter"),
        # The ceiling has to clear a whole city, not a sample of one: the map
        # plots every stop at once, and a cap below the feed's stop count
        # silently draws a partial network. Bengaluru alone has 9,948.
        limit: int = Query(500, ge=1, le=50_000),
    ) -> list[StopOut]:
        needle = q.lower() if q else None
        out: list[StopOut] = []
        for index, stop in enumerate(engine.timetable.stops):
            if needle and needle not in stop.name.lower() and needle not in stop.stop_id.lower():
                continue
            out.append(engine.stop_out(index))
            if len(out) >= limit:
                break
        return out

    @app.get("/journey", response_model=JourneyResponse)
    def journey(
        engine: EngineDep,
        origin: str = Query(..., alias="from", description="origin stop_id"),
        destination: str = Query(..., alias="to", description="destination stop_id"),
        time: str = Query("08:00:00", description="earliest departure, HH:MM[:SS]"),
        change_time: int = Query(120, ge=0, le=3600, description="seconds per change"),
        max_walk: int | None = Query(None, ge=0, le=3600),
    ) -> JourneyResponse:
        source = engine.resolve(origin, "origin")
        target = engine.resolve(destination, "destination")
        departure = _parse_time(time)

        # One one-to-all run answers both the fastest journey and the
        # transfer-count alternatives; target pruning is left off because the
        # alternatives read other rounds of the same table.
        result = engine.router(change_time).run(
            source, departure, max_walk_seconds=max_walk
        )
        fastest = result.journey(target)
        alternatives = [
            _journey_out(engine, j)
            for j in result.journeys_by_transfers(target)
            if fastest is None or j.legs != fastest.legs
        ]
        return JourneyResponse(
            origin=engine.stop_out(source),
            destination=engine.stop_out(target),
            requested_departure=format_gtfs_time(departure),
            fastest=_journey_out(engine, fastest) if fastest else None,
            alternatives=alternatives,
        )

    @app.get("/isochrone", response_model=IsochroneResponse)
    def isochrone(
        engine: EngineDep,
        origin: str = Query(..., alias="from"),
        time: str = Query("08:00:00"),
        minutes: int = Query(30, ge=1, le=600),
        change_time: int = Query(120, ge=0, le=3600),
        max_walk: int | None = Query(None, ge=0, le=3600),
    ) -> IsochroneResponse:
        source = engine.resolve(origin, "origin")
        departure = _parse_time(time)
        result = engine.router(change_time).run(
            source, departure, max_walk_seconds=max_walk
        )

        reached = []
        for index in result.reachable(minutes * 60).tolist():
            arrival = int(result.best[index])
            stop = engine.timetable.stops[index]
            reached.append(
                ReachedStop(
                    stop_id=stop.stop_id,
                    name=stop.name,
                    lat=stop.lat,
                    lon=stop.lon,
                    arrival=format_gtfs_time(arrival),
                    minutes=round((arrival - departure) / 60),
                )
            )
        reached.sort(key=lambda s: s.minutes)
        return IsochroneResponse(
            origin=engine.stop_out(source),
            departure=format_gtfs_time(departure),
            budget_minutes=minutes,
            reached=reached,
            unreachable=int(np.count_nonzero(result.best >= NEVER)),
        )

    # Mounted last so it cannot shadow an API route.
    if WEB_ROOT.is_dir():
        app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")

    return app


app = create_app()
