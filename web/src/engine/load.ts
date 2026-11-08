/**
 * Fetch and decompress the three bundle files.
 *
 * The files are gzipped by the build rather than by the host, because a
 * static host may or may not compress an unknown binary type and the size of
 * the download should not depend on which. That leaves one hazard worth
 * guarding: if a host *does* advertise `Content-Encoding: gzip` on a `.gz`
 * file, the browser unwraps it before we see it, and decompressing again
 * fails. So the bytes are sniffed for the gzip magic number rather than
 * assumed to be compressed -- two bytes of inspection instead of a
 * deployment that works on one host and not the next.
 */

import {
  decodeBundle,
  type BundleManifest,
  type BundleStrings,
} from "./bundle.ts";
import type { Timetable } from "./timetable.ts";

const GZIP_MAGIC = [0x1f, 0x8b];

async function gunzipIfNeeded(bytes: Uint8Array): Promise<Uint8Array> {
  if (bytes[0] !== GZIP_MAGIC[0] || bytes[1] !== GZIP_MAGIC[1]) return bytes;
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(
    new DecompressionStream("gzip"),
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export interface LoadProgress {
  /** Bytes of compressed payload received so far. */
  received: number;
  /** Total compressed bytes, when the server sent a length. */
  total: number | null;
  phase: "manifest" | "timetable" | "strings" | "decoding";
}

async function fetchBytes(
  url: string,
  onProgress?: (received: number, total: number | null) => void,
): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status} ${response.statusText}`);
  }
  const header = response.headers.get("content-length");
  const total = header ? Number(header) : null;

  if (!response.body || !onProgress) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

async function fetchJson<T>(url: string): Promise<T> {
  const bytes = await gunzipIfNeeded(await fetchBytes(url));
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export interface LoadedFeed {
  timetable: Timetable;
  manifest: BundleManifest;
  /** Compressed bytes actually transferred, for the console to report. */
  transferred: number;
  /** Milliseconds from the first request to a routable timetable. */
  elapsedMs: number;
}

/**
 * Load a feed bundle from `baseUrl` (a directory, relative or absolute).
 *
 * Relative by default and relative on purpose: a GitHub Pages project site is
 * served from `/<repo>/`, and an absolute path is a URL that works in
 * development and 404s the moment it is published.
 */
export async function loadFeed(
  baseUrl = "feed",
  onProgress?: (progress: LoadProgress) => void,
): Promise<LoadedFeed> {
  const started = performance.now();
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  let transferred = 0;

  onProgress?.({ received: 0, total: null, phase: "manifest" });
  const manifest = await fetchJson<BundleManifest>(`${base}manifest.json.gz`);

  onProgress?.({ received: 0, total: null, phase: "timetable" });
  const binary = await gunzipIfNeeded(
    await fetchBytes(`${base}timetable.bin.gz`, (received, total) => {
      transferred = received;
      onProgress?.({ received, total, phase: "timetable" });
    }),
  );

  onProgress?.({ received: transferred, total: transferred, phase: "strings" });
  const strings = await fetchJson<BundleStrings>(`${base}strings.json.gz`);

  onProgress?.({ received: transferred, total: transferred, phase: "decoding" });
  // The views the decoder takes must sit on an ArrayBuffer whose start is the
  // start of the payload; a Uint8Array from a stream can be a window onto a
  // larger buffer, so this slices rather than hands over `.buffer`.
  const buffer = binary.buffer.slice(
    binary.byteOffset,
    binary.byteOffset + binary.byteLength,
  ) as ArrayBuffer;
  const timetable = decodeBundle(manifest, buffer, strings);

  return {
    timetable,
    manifest,
    transferred,
    elapsedMs: performance.now() - started,
  };
}
