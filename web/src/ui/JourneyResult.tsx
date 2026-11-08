/**
 * The journey, as a duration and then a timetable.
 *
 * The big number is minutes, because that is the unit the project argues a
 * city should be measured in. Underneath it the legs are set as an actual
 * column of departure times -- monospace, tabular figures, one row per
 * boarding -- rather than as a list of cards, because a column of times is
 * what a passenger reads and the alignment is the whole point of it.
 */

import type { JourneyOut } from "../engine/api.ts";
import { rampColor } from "./ramp.ts";

const hhmm = (clock: string) => clock.slice(0, 5);

function shape(journey: JourneyOut): string {
  const changes =
    journey.transfers === 0
      ? "direct"
      : `${journey.transfers} change${journey.transfers === 1 ? "" : "s"}`;
  const walk = journey.walk_minutes > 0 ? `, ${journey.walk_minutes} min walking` : "";
  return `${changes}${walk}`;
}

interface Props {
  journey: JourneyOut;
  alternatives: JourneyOut[];
  destinationName: string;
  chosen: number;
  onChoose: (at: number) => void;
}

export function JourneyResult({
  journey,
  alternatives,
  destinationName,
  chosen,
  onChoose,
}: Props) {
  let elapsed = 0;
  const rows = journey.legs.map((leg, at) => {
    elapsed += leg.seconds;
    const track = rampColor(Math.round(elapsed / 60));
    const interchange = at > 0 && journey.legs[at - 1].kind === "ride" && leg.kind === "ride";
    return { leg, track, interchange, key: `${leg.from_stop}-${at}` };
  });

  return (
    <div className="result">
      <div className="verdict">
        <p className="verdict__time">
          {journey.duration_minutes}
          <small>min</small>
        </p>
        <p className="verdict__shape">{shape(journey)}</p>
        <p className="verdict__clock">
          leave {hhmm(journey.departure)} · arrive {hhmm(journey.arrival)}
        </p>
      </div>

      <ol className="legs">
        {rows.map(({ leg, track, interchange, key }) => (
          <li className="leg" key={key} style={{ ["--track" as string]: track }}>
            <span className={`leg__time${leg.kind === "walk" ? " leg__time--blank" : ""}`}>
              {leg.kind === "ride" ? hhmm(leg.departure ?? "") : "walk"}
            </span>
            <span
              className={
                "leg__spine" +
                (leg.kind === "walk" ? " leg__spine--walk" : "") +
                (interchange ? " leg__spine--interchange" : "")
              }
            />
            <span className="leg__body">
              <span className="leg__stop">{leg.from_name}</span>
              <span className="leg__detail">
                {/*
                  The plaque is route_name, never route_id. The id is the
                  feed's internal key ("2130"); the name is what is painted
                  on the front of the bus ("VW-226KBS"), and only one of
                  those helps anyone board.
                */}
                {leg.kind === "ride" ? (
                  <>
                    <span className="plaque">{leg.route_name}</span>{" "}
                    {Math.max(1, Math.round(leg.seconds / 60))} min to {leg.to_name}
                  </>
                ) : (
                  <>
                    Walk {Math.max(1, Math.round(leg.seconds / 60))} min to {leg.to_name}
                    {/*
                      A walk that starts and ends at the same name is not a
                      bug and is the single most confusing thing this feed
                      produces. BMTC gives each direction of a road its own
                      stop -- 9,948 ids against about 5,200 distinct names --
                      so crossing the street is a walk between two stops
                      called the same thing. The codes are what tell them
                      apart, so they are shown exactly when they are needed.
                    */}
                    {leg.from_name === leg.to_name && (
                      <> — the other side of the road, stop {leg.to_stop}</>
                    )}
                  </>
                )}
              </span>
            </span>
          </li>
        ))}
        <li className="leg leg--end" style={{ ["--track" as string]: "#e8f0ee" }}>
          <span className="leg__time">{hhmm(journey.arrival)}</span>
          <span className="leg__spine" />
          <span className="leg__body">
            <span className="leg__stop">{destinationName}</span>
          </span>
        </li>
      </ol>

      {alternatives.length > 0 && (
        <div className="alts">
          <p className="alts__head">
            Fewer changes, if you would rather not transfer
          </p>
          <button
            className="alt"
            type="button"
            aria-pressed={chosen === -1}
            onClick={() => onChoose(-1)}
          >
            <span className="alt__time">{journey.duration_minutes} min</span>
            <span className="alt__shape">{shape(journey)}</span>
          </button>
          {alternatives.map((option, at) => (
            <button
              className="alt"
              type="button"
              key={`${option.arrival}-${option.transfers}-${at}`}
              aria-pressed={chosen === at}
              onClick={() => onChoose(at)}
            >
              <span className="alt__time">{option.duration_minutes} min</span>
              <span className="alt__shape">{shape(option)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
