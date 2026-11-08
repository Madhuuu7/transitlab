/**
 * The time ramp: the one accent scale in the console.
 *
 * Mint for the first ten minutes, ice blue to twenty, then two depths of blue
 * beyond. It colours the isochrone, the journey's own legs, the histogram and
 * the legend, so a colour in this interface always answers the same question:
 * how long does it take to get there. That is also the premise of the project,
 * which is why the accent scale is a time scale and not a brand palette.
 */

export interface RampStep {
  /** Upper bound in minutes, inclusive. */
  upTo: number;
  color: string;
  label: string;
}

export const RAMP: RampStep[] = [
  { upTo: 10, color: "#5fd6a0", label: "0" },
  { upTo: 20, color: "#7fc8e8", label: "10" },
  { upTo: 30, color: "#4a8fb8", label: "20" },
  { upTo: Number.POSITIVE_INFINITY, color: "#2a5570", label: "30+" },
];

export function rampColor(minutes: number): string {
  for (const step of RAMP) if (minutes <= step.upTo) return step.color;
  return RAMP[RAMP.length - 1].color;
}

/**
 * The ramp as a MapLibre `step` expression, so thousands of stops are coloured
 * by the GPU from one `minutes` property rather than by building a feature per
 * colour on the CPU.
 */
export function rampExpression(): unknown[] {
  const expression: unknown[] = ["step", ["get", "minutes"], RAMP[0].color];
  for (let i = 0; i < RAMP.length - 1; i += 1) {
    expression.push(RAMP[i].upTo, RAMP[i + 1].color);
  }
  return expression;
}

/** Counts per ramp band, for the histogram. */
export function bandCounts(minutes: number[]): Array<{ step: RampStep; n: number }> {
  const counts = RAMP.map((step) => ({ step, n: 0 }));
  for (const value of minutes) {
    for (let i = 0; i < RAMP.length; i += 1) {
      if (value <= RAMP[i].upTo) {
        counts[i].n += 1;
        break;
      }
    }
  }
  return counts;
}
