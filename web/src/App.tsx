/**
 * The console.
 *
 * There is no backend. The timetable is downloaded once and every query is
 * answered in this tab by the same RAPTOR implementation the Python server
 * runs, checked against it in CI on 49 journeys and 6 isochrones over this
 * feed. So nothing on this page is a recording, and the footer says what it
 * is actually doing rather than making a claim about a server that is not
 * there.
 *
 * Queries run on the main thread on purpose. A one-to-all run over the whole
 * city is around 40 ms, which is below the threshold where a spinner helps
 * and well below where a worker would earn the complexity of moving the
 * timetable into one.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { LocalEngine, type IsochroneResponse, type JourneyResponse } from "./engine/api.ts";
import { loadFeed, type LoadProgress } from "./engine/load.ts";
import type { Timetable } from "./engine/timetable.ts";
import { JourneyResult } from "./ui/JourneyResult.tsx";
import { MapView } from "./ui/MapView.tsx";
import { ReachResult } from "./ui/ReachResult.tsx";
import { StopCombo } from "./ui/StopCombo.tsx";

type Mode = "journey" | "reach";

interface Loaded {
  engine: LocalEngine;
  timetable: Timetable;
  transferred: number;
  decodeMs: number;
}

/** The journey the README quotes, so the page opens on something real. */
const OPENING = { from: "Majestic", to: "Electronic City" };

function findByName(timetable: Timetable, needle: string): string | null {
  const lowered = needle.toLowerCase();
  for (const stop of timetable.stops) {
    if (stop.name.toLowerCase().includes(lowered)) return stop.stopId;
  }
  return null;
}

function Boot({ progress, error }: { progress: LoadProgress | null; error: string | null }) {
  const received = progress?.received ?? 0;
  const total = progress?.total ?? null;
  const pct = total ? Math.min(100, (received / total) * 100) : received > 0 ? 40 : 4;

  return (
    <div className="boot">
      <div className="boot__inner">
        <h1 className="boot__name">
          transit<span>lab</span>
        </h1>
        <p className="boot__what">
          Fetching Bengaluru's bus timetable — 9,948 stops and 1.5 million stop times.
          It routes here in your browser, so this download happens once and nothing
          after it touches a server.
        </p>
        {error ? (
          <div className="boot__error" role="alert">
            Could not load the feed: {error}
            <br />
            <br />
            If you are running this locally, build the bundle first:{" "}
            <code>python scripts/make_web_bundle.py data/bmtc.npz web/public/feed</code>
          </div>
        ) : (
          <>
            <div
              className="boot__meter"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(pct)}
            >
              <div className="boot__fill" style={{ width: `${pct}%` }} />
            </div>
            <p className="boot__read">
              <span>
                {progress?.phase === "decoding"
                  ? "decoding the timetable"
                  : progress?.phase === "strings"
                    ? "reading stop names"
                    : "downloading"}
              </span>
              <span>
                {(received / 1e6).toFixed(2)}
                {total ? ` / ${(total / 1e6).toFixed(2)}` : ""} MB
              </span>
            </p>
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [progress, setProgress] = useState<LoadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<Mode>("journey");
  const [from, setFrom] = useState<string | null>(null);
  const [to, setTo] = useState<string | null>(null);
  const [time, setTime] = useState("08:00");
  const [changeTime, setChangeTime] = useState(120);
  const [budget, setBudget] = useState(30);

  const [journey, setJourney] = useState<JourneyResponse | null>(null);
  const [chosen, setChosen] = useState(-1);
  const [reach, setReach] = useState<IsochroneResponse | null>(null);
  const [queryMs, setQueryMs] = useState<number | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadFeed("feed", (update) => {
      if (!cancelled) setProgress(update);
    })
      .then((feed) => {
        if (cancelled) return;
        const engine = new LocalEngine(feed.timetable, feed.manifest.feed);
        setLoaded({
          engine,
          timetable: feed.timetable,
          transferred: feed.transferred,
          decodeMs: feed.elapsedMs,
        });
        setFrom(findByName(feed.timetable, OPENING.from));
        setTo(findByName(feed.timetable, OPENING.to));
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const run = useCallback(() => {
    if (!loaded || !from) return;
    setQueryError(null);
    const started = performance.now();
    try {
      if (mode === "journey") {
        if (!to) return;
        setJourney(loaded.engine.journey({ from, to, time, changeTime }));
        setReach(null);
        setChosen(-1);
      } else {
        setReach(loaded.engine.isochrone({ from, time, minutes: budget, changeTime }));
        setJourney(null);
      }
      setQueryMs(performance.now() - started);
    } catch (cause: unknown) {
      setQueryError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [loaded, from, to, time, changeTime, budget, mode]);

  // Route once the feed and the opening stops are in place, so the page
  // arrives with an answer on it rather than an empty form.
  const opened = Boolean(loaded && from && to && !journey && !reach);
  useEffect(() => {
    if (opened) run();
  }, [opened, run]);

  const shown = useMemo(() => {
    if (!journey?.fastest) return null;
    if (chosen === -1) return journey.fastest;
    return journey.alternatives[chosen] ?? journey.fastest;
  }, [journey, chosen]);

  if (!loaded) return <Boot progress={progress} error={error} />;

  const feed = loaded.engine.feed();
  const stops = loaded.timetable.stops;

  return (
    <div className="shell">
      <div className="rail">
        <div className="rail__scroll">
          <header className="masthead">
            <h1 className="masthead__name">
              transit<span>lab</span>
            </h1>
            <p className="masthead__where">
              Bengaluru buses, routed in this tab
            </p>
          </header>

          <div className="modes">
            <button
              className="modes__btn"
              type="button"
              aria-pressed={mode === "journey"}
              onClick={() => setMode("journey")}
            >
              Plan a journey
            </button>
            <button
              className="modes__btn"
              type="button"
              aria-pressed={mode === "reach"}
              onClick={() => setMode("reach")}
            >
              How far in time
            </button>
          </div>

          <form
            className="query"
            onSubmit={(event) => {
              event.preventDefault();
              run();
            }}
          >
            <StopCombo
              id="from"
              label={mode === "journey" ? "From" : "Starting from"}
              stops={stops}
              value={from}
              onChange={setFrom}
            />
            {mode === "journey" && (
              <>
                <StopCombo
                  id="to"
                  label="To"
                  stops={stops}
                  value={to}
                  onChange={setTo}
                />
                <button
                  className="swap"
                  type="button"
                  onClick={() => {
                    setFrom(to);
                    setTo(from);
                  }}
                >
                  Reverse the trip
                </button>
              </>
            )}

            <div className="field__row">
              <div className="field">
                <label className="field__label" htmlFor="time">
                  Leave at
                </label>
                <input
                  id="time"
                  className="control control--mono"
                  type="time"
                  value={time}
                  onChange={(event) => setTime(event.target.value)}
                />
              </div>
              <div className="field">
                <label className="field__label" htmlFor="change">
                  Time to change
                </label>
                <select
                  id="change"
                  className="control control--mono"
                  value={changeTime}
                  onChange={(event) => setChangeTime(Number(event.target.value))}
                >
                  <option value={0}>none</option>
                  <option value={60}>1 min</option>
                  <option value={120}>2 min</option>
                  <option value={300}>5 min</option>
                </select>
              </div>
            </div>

            {mode === "reach" && (
              <div className="field">
                <label className="field__label" htmlFor="budget">
                  Within
                </label>
                <select
                  id="budget"
                  className="control control--mono"
                  value={budget}
                  onChange={(event) => setBudget(Number(event.target.value))}
                >
                  {[15, 30, 45, 60, 90].map((minutes) => (
                    <option key={minutes} value={minutes}>
                      {minutes} minutes
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button className="go" type="submit" disabled={!from || (mode === "journey" && !to)}>
              {mode === "journey" ? "Find the fastest way" : "Show what is reachable"}
            </button>
          </form>

          {queryError && (
            <div className="boot__error" role="alert" style={{ margin: "0 22px 18px" }}>
              {queryError}
            </div>
          )}

          {mode === "journey" && journey && !journey.fastest && (
            <div className="result">
              <div className="verdict">
                <p className="verdict__shape">
                  No journey from {journey.origin.name} to {journey.destination.name} after{" "}
                  {journey.requested_departure.slice(0, 5)}. The schedule may not run at this
                  hour, or the two stops may sit on parts of the network that never connect.
                  Try a morning departure.
                </p>
              </div>
            </div>
          )}

          {mode === "journey" && journey && shown && (
            <JourneyResult
              journey={shown}
              alternatives={journey.alternatives}
              destinationName={journey.destination.name}
              chosen={chosen}
              onChoose={setChosen}
            />
          )}

          {mode === "reach" && reach && (
            <ReachResult result={reach} totalStops={feed.stops} />
          )}
        </div>

        <footer className="ledger">
          <div className="ledger__row">
            <span>no server · routed in this tab</span>
            <span>
              <b>{queryMs === null ? "—" : `${queryMs.toFixed(0)} ms`}</b> last query
            </span>
          </div>
          <div className="ledger__row">
            <span>
              {feed.feed} · <b>{feed.stops.toLocaleString("en-IN")}</b> stops ·{" "}
              <b>{(feed.stop_times / 1e6).toFixed(2)}M</b> stop times
            </span>
          </div>
          <div className="ledger__row">
            <span>
              <b>{(loaded.transferred / 1e6).toFixed(2)} MB</b> downloaded, decoded in{" "}
              <b>{loaded.decodeMs.toFixed(0)} ms</b>
            </span>
            <span>
              <a href="https://github.com/Madhuuu7/transitlab">source</a>
            </span>
          </div>
        </footer>
      </div>

      <MapView
        stops={stops}
        timetable={loaded.timetable}
        journey={mode === "journey" ? shown : null}
        origin={mode === "reach" ? (reach?.origin ?? null) : (journey?.origin ?? null)}
        destination={journey?.destination ?? null}
        reached={mode === "reach" && reach ? reach.reached : null}
      />
    </div>
  );
}
