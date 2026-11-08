/**
 * A stop picker over 9,948 stops.
 *
 * Which rules out a `<select>`: the first version of this console rendered
 * every stop as an option and the browser hung on it. It also rules out
 * relying on the names alone -- BMTC models each direction of a road as its
 * own stop, so there are 9,960 ids against about 5,200 distinct names, and
 * "MG Road" is six different stops. The code is shown beside every name for
 * that reason, not as decoration.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { Stop } from "../engine/timetable.ts";

const SHOWN = 60;

interface Props {
  label: string;
  stops: Stop[];
  value: string | null;
  onChange: (stopId: string) => void;
  id: string;
}

function nameOf(stops: Stop[], index: Map<string, number>, id: string | null): string {
  if (!id) return "";
  const at = index.get(id);
  return at === undefined ? "" : stops[at].name;
}

/** The matched run of the name, so a long list is scannable. */
function Highlight({ text, needle }: { text: string; needle: string }) {
  if (!needle) return <span>{text}</span>;
  const at = text.toLowerCase().indexOf(needle.toLowerCase());
  if (at < 0) return <span>{text}</span>;
  return (
    <span>
      {text.slice(0, at)}
      <b>{text.slice(at, at + needle.length)}</b>
      {text.slice(at + needle.length)}
    </span>
  );
}

export function StopCombo({ label, stops, value, onChange, id }: Props) {
  const index = useMemo(
    () => new Map(stops.map((stop, i) => [stop.stopId, i])),
    [stops],
  );
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return stops.slice(0, SHOWN);
    const out: Stop[] = [];
    for (const stop of stops) {
      if (
        stop.name.toLowerCase().includes(needle) ||
        stop.stopId.toLowerCase().includes(needle)
      ) {
        out.push(stop);
        if (out.length >= SHOWN) break;
      }
    }
    return out;
  }, [query, stops]);

  useEffect(() => {
    if (!open) return;
    const away = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);

  const choose = (stop: Stop) => {
    onChange(stop.stopId);
    setQuery("");
    setOpen(false);
  };

  return (
    <div className="field" ref={box}>
      <label className="field__label" htmlFor={id}>
        {label}
      </label>
      <div className="combo">
        <input
          id={id}
          className="control"
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-autocomplete="list"
          aria-activedescendant={open ? `${id}-opt-${active}` : undefined}
          value={open ? query : nameOf(stops, index, value)}
          placeholder="Search a stop or code"
          onFocus={() => {
            setOpen(true);
            setQuery("");
            setActive(0);
          }}
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
            setActive(0);
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setOpen(true);
              setActive((at) => Math.min(at + 1, matches.length - 1));
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              setActive((at) => Math.max(at - 1, 0));
            } else if (event.key === "Enter" && open && matches[active]) {
              event.preventDefault();
              choose(matches[active]);
            } else if (event.key === "Escape") {
              setOpen(false);
            }
          }}
        />
        {open && (
          <ul className="combo__list" id={`${id}-list`} role="listbox">
            {matches.length === 0 && (
              <li className="combo__empty">
                No stop matches that. The feed transliterates: try “Shivajinagara”,
                “White Field”.
              </li>
            )}
            {matches.map((stop, at) => (
              <li
                key={stop.stopId}
                id={`${id}-opt-${at}`}
                className="combo__option"
                role="option"
                aria-selected={at === active}
                onMouseEnter={() => setActive(at)}
                onMouseDown={(event) => {
                  event.preventDefault();
                  choose(stop);
                }}
              >
                <Highlight text={stop.name} needle={query.trim()} />
                <span className="combo__code">{stop.stopId}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
