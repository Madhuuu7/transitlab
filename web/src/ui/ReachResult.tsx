/**
 * How far you can get, as a count and a histogram of arrival times.
 *
 * The histogram is the finding, not an ornament: a city's reachable set is
 * not spread evenly over the budget, it arrives in waves set by where the
 * frequent corridors run. The bars use the same ramp as the map, so a band
 * and the dots it describes are the same colour.
 */

import type { IsochroneResponse } from "../engine/api.ts";
import { bandCounts } from "./ramp.ts";

interface Props {
  result: IsochroneResponse;
  totalStops: number;
}

export function ReachResult({ result, totalStops }: Props) {
  // Bands the budget cannot reach are dropped rather than drawn empty: a row
  // reading "30+ min 0" under a 30-minute query describes the question, not
  // the city.
  const counts = bandCounts(result.reached.map((stop) => stop.minutes)).filter(
    (_band, at, all) =>
      at === 0 || Number(all[at - 1].step.upTo) < result.budget_minutes,
  );
  const widest = Math.max(1, ...counts.map((band) => band.n));
  const share = ((result.reached.length / totalStops) * 100).toFixed(1);

  return (
    <div className="reach">
      <p className="reach__count">{result.reached.length.toLocaleString("en-IN")}</p>
      <p className="reach__of">
        stops within {result.budget_minutes} minutes of {result.origin.name}, leaving at{" "}
        {result.departure.slice(0, 5)} — {share}% of the network. {result.unreachable.toLocaleString("en-IN")}{" "}
        stops are not reachable at all at this hour.
      </p>

      <div className="bands">
        {counts.map((band) => (
          <div className="band" key={band.step.label}>
            <span className="band__range">
              {Number.isFinite(band.step.upTo)
                ? `${band.step.label}–${Math.min(Number(band.step.upTo), result.budget_minutes)} min`
                : `${band.step.label.replace("+", "")}–${result.budget_minutes} min`}
            </span>
            <span
              className="band__bar"
              style={{
                ["--band" as string]: band.step.color,
                width: `${(band.n / widest) * 100}%`,
              }}
            />
            <span className="band__n">{band.n.toLocaleString("en-IN")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
