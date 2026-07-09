import { afterEach, describe, expect, it, vi } from "vitest";

import { extractFrames, parseChatEvent, streamChat } from "../chatStream";

describe("extractFrames", () => {
  it("extracts a single complete frame", () => {
    const { frames, remainder } = extractFrames("event: meta\ndata: {}\n\n");

    expect(frames).toEqual(["event: meta\ndata: {}"]);
    expect(remainder).toBe("");
  });

  it("buffers an incomplete frame with no trailing blank line", () => {
    const { frames, remainder } = extractFrames("event: meta\ndata: {");

    expect(frames).toEqual([]);
    expect(remainder).toBe("event: meta\ndata: {");
  });

  it("extracts multiple frames arriving in one chunk", () => {
    const buffer = "event: meta\ndata: {}\n\nevent: delta\ndata: {\"text\":\"hi\"}\n\n";

    const { frames, remainder } = extractFrames(buffer);

    expect(frames).toEqual(["event: meta\ndata: {}", 'event: delta\ndata: {"text":"hi"}']);
    expect(remainder).toBe("");
  });

  it("reassembles a frame split across two network chunks", () => {
    // A real fetch stream can split a frame anywhere, including mid-JSON-token.
    const chunk1 = 'event: delta\ndata: {"te';
    const first = extractFrames(chunk1);
    expect(first.frames).toEqual([]);
    expect(first.remainder).toBe(chunk1);

    const chunk2 = first.remainder + 'xt":"hello"}\n\nevent: done\ndata: {}\n\n';
    const second = extractFrames(chunk2);

    expect(second.frames).toEqual(['event: delta\ndata: {"text":"hello"}', "event: done\ndata: {}"]);
    expect(second.remainder).toBe("");
  });
});

describe("parseChatEvent", () => {
  it("parses a meta event", () => {
    const event = parseChatEvent(
      'event: meta\ndata: {"route":"tools","tools":["iss_now"],"iterations":2}',
    );

    expect(event).toEqual({
      type: "meta",
      data: { route: "tools", tools: ["iss_now"], iterations: 2 },
    });
  });

  it("parses a sequence of interleaved event types in order", () => {
    const frames = [
      'event: meta\ndata: {"route":"direct","tools":[],"iterations":1}',
      "event: sources\ndata: []",
      'event: delta\ndata: {"text":"Hello "}',
      'event: delta\ndata: {"text":"world."}',
      'event: done\ndata: {"total_latency_ms":12.5,"token_usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"cited_sources":[],"invalid_citations":[]}',
    ];

    const events = frames.map(parseChatEvent);

    expect(events.map((e) => e?.type)).toEqual(["meta", "sources", "delta", "delta", "done"]);
    expect(events[2]).toEqual({ type: "delta", data: { text: "Hello " } });
    expect(events[4]?.data).toMatchObject({ total_latency_ms: 12.5 });
  });

  it("parses a multi-line data payload with embedded newlines rejoined", () => {
    // json.dumps never emits raw newlines, but data: can legally repeat across
    // lines per the SSE spec - the parser should still concatenate them.
    const event = parseChatEvent('event: error\ndata: {"message":\ndata: "boom"}');

    expect(event).toEqual({ type: "error", data: { message: "boom" } });
  });

  it("returns null for a frame with malformed JSON", () => {
    expect(parseChatEvent("event: meta\ndata: not valid json")).toBeNull();
  });

  it("returns null for a frame with no event line", () => {
    expect(parseChatEvent("data: {}")).toBeNull();
  });

  it("returns null for an unrecognized event name", () => {
    expect(parseChatEvent('event: mystery\ndata: {"x":1}')).toBeNull();
  });

  it("returns null for a completely empty frame", () => {
    expect(parseChatEvent("")).toBeNull();
  });
});

describe("streamChat", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function fakeResponseFromStream(stream: ReadableStream<Uint8Array>): Response {
    return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
  }

  it("yields typed events parsed from a streamed response body", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('event: meta\ndata: {"route":"direct","tools":[],"iterations":1}\n\n'));
        controller.enqueue(encoder.encode('event: delta\ndata: {"text":"hi"}\n\n'));
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(fakeResponseFromStream(stream)),
    );

    const events = [];
    for await (const event of streamChat("q", { apiUrl: "http://api.test" })) {
      events.push(event);
    }

    expect(events.map((e) => e.type)).toEqual(["meta", "delta"]);
  });

  it("throws on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 500, statusText: "Internal Server Error" })),
    );

    await expect(async () => {
      const generator = streamChat("q", { apiUrl: "http://api.test" });
      await generator.next();
    }).rejects.toThrow(/500/);
  });

  it("stops iterating when the abort signal fires mid-stream", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('event: meta\ndata: {"route":"direct","tools":[],"iterations":1}\n\n'));
        // Deliberately never closes - simulates a slow/ongoing stream so the
        // abort check (not stream completion) is what ends iteration.
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(fakeResponseFromStream(stream)),
    );

    const controller = new AbortController();
    const received: string[] = [];

    await expect(async () => {
      for await (const event of streamChat("q", { apiUrl: "http://api.test", signal: controller.signal })) {
        received.push(event.type);
        controller.abort();
      }
    }).rejects.toMatchObject({ name: "AbortError" });

    expect(received).toEqual(["meta"]);
  });
});
