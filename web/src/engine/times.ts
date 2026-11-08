/**
 * GTFS time handling, ported from `transitlab.gtfs.times`.
 *
 * Seconds since the start of the service day, never reduced modulo a day:
 * GTFS writes a trip that leaves at 23:50 and arrives after midnight as
 * `24:20:00`, and wrapping that would turn a 30-minute ride into a 23.5-hour
 * one. See the Python module for the full reasoning.
 */

/** Sentinel for "unreachable", with headroom so `arrival + change` cannot overflow. */
export const NEVER = 2_000_000_000;

export const SECONDS_PER_DAY = 86_400;

/**
 * Parse `HH:MM:SS` (or `HH:MM`) into seconds since the service day started.
 *
 * Hours above 23 are valid and preserved. Single-digit hours are accepted
 * because real feeds contain them despite the spec. Accepting `HH:MM` matches
 * what the HTTP API accepts, not what the parser underneath it does -- the
 * console's time input produces exactly that shape.
 */
export function parseGtfsTime(value: string): number {
  const text = value.trim();
  if (!text) throw new Error("empty GTFS time");

  const parts = text.split(":");
  if (parts.length === 2) parts.push("00");
  if (parts.length !== 3) throw new Error(`malformed GTFS time: ${value}`);

  const [hours, minutes, seconds] = parts.map((part) => {
    if (!/^\d+$/.test(part.trim())) throw new Error(`malformed GTFS time: ${value}`);
    return Number(part);
  });
  if (!(minutes < 60 && seconds < 60)) {
    throw new Error(`minutes/seconds out of range: ${value}`);
  }
  return hours * 3600 + minutes * 60 + seconds;
}

/** Inverse of {@link parseGtfsTime}, keeping hours >= 24 intact. */
export function formatGtfsTime(seconds: number): string {
  if (seconds >= NEVER) return "--:--:--";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}`;
}

/**
 * Python's `round`: half-way values go to the nearest **even** integer.
 *
 * This is not pedantry. The API reports durations as `round(seconds / 60)`,
 * so a 150-second walk is 2 minutes in Python and would be 3 with
 * `Math.round`. One such leg is enough to make the browser disagree with the
 * server about a journey both of them routed identically, and the parity
 * suite would report it as a routing bug.
 */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const remainder = value - floor;
  if (remainder > 0.5) return floor + 1;
  if (remainder < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Minutes, rounded the way the API rounds them. */
export function toMinutes(seconds: number): number {
  return roundHalfEven(seconds / 60);
}
