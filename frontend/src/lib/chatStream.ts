/**
 * Typed SSE client for POST /chat.
 *
 * EventSource doesn't support POST, so this hand-rolls SSE parsing over a
 * fetch() ReadableStream - mirroring scripts/chat_client.py, the backend's
 * own reference consumer. Frames are separated by a blank line ("\n\n") and
 * may be split across network chunks, so incomplete frames are buffered
 * until the rest arrives (see extractFrames).
 */

import type { ChatEvent } from "./types";

/**
 * Pulls complete "event: ...\ndata: ..." frames out of a growing buffer.
 * Returns the complete frames found so far and whatever incomplete tail
 * should be kept and prepended to the next chunk.
 */
export function extractFrames(buffer: string): { frames: string[]; remainder: string } {
  const frames: string[] = [];
  let rest = buffer;
  let separatorIndex = rest.indexOf("\n\n");
  while (separatorIndex !== -1) {
    frames.push(rest.slice(0, separatorIndex));
    rest = rest.slice(separatorIndex + 2);
    separatorIndex = rest.indexOf("\n\n");
  }
  return { frames, remainder: rest };
}

/** Splits a single frame into its event name and raw (possibly multi-line) data. */
function parseFrameLines(frame: string): { event: string; data: string } | null {
  let event: string | null = null;
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }
  if (!event) return null;
  return { event, data: dataLines.join("\n") };
}

const KNOWN_EVENTS = new Set(["meta", "sources", "delta", "done", "error"]);

/**
 * Parses one raw SSE frame into a typed ChatEvent, or null if the frame is
 * malformed (no event line, invalid JSON, or an unrecognized event name) -
 * malformed frames are dropped rather than crashing the stream.
 */
export function parseChatEvent(frame: string): ChatEvent | null {
  const parsed = parseFrameLines(frame);
  if (!parsed || !KNOWN_EVENTS.has(parsed.event)) return null;

  try {
    const data = parsed.data ? JSON.parse(parsed.data) : null;
    return { type: parsed.event, data } as ChatEvent;
  } catch {
    return null;
  }
}

export interface StreamChatOptions {
  apiUrl: string;
  signal?: AbortSignal;
}

/**
 * POSTs the query to /chat and yields typed events as they stream in.
 * Throws on a non-2xx response, a network failure, or an abort (the standard
 * DOMException with name "AbortError" - callers should check `err.name` to
 * distinguish a deliberate cancel from a real failure).
 */
export async function* streamChat(
  query: string,
  { apiUrl, signal }: StreamChatOptions,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${apiUrl}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error("Response had no body to stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const { frames, remainder } = extractFrames(buffer);
      buffer = remainder;

      for (const frame of frames) {
        const event = parseChatEvent(frame);
        if (event) yield event;
      }
    }

    // Flush the decoder and check for a final frame with no trailing blank line.
    buffer += decoder.decode();
    if (buffer.trim()) {
      const event = parseChatEvent(buffer);
      if (event) yield event;
    }
  } finally {
    reader.releaseLock();
  }
}
