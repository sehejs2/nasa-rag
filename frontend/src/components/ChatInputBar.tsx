"use client";

import { useRef, useState, type KeyboardEvent } from "react";

interface ChatInputBarProps {
  onSend: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  showSamples: boolean;
}

const SAMPLE_QUERIES = [
  "What did Webb discover about the Cigar Galaxy?",
  "Where is the ISS right now?",
  "Show me a recent Perseverance photo and its mission history.",
  "What is today's astronomy picture of the day?",
];

const MAX_TEXTAREA_HEIGHT_PX = 160;

export function ChatInputBar({ onSend, onStop, isStreaming, showSamples }: ChatInputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function resizeTextarea() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`;
  }

  function handleSend(query?: string) {
    const trimmed = (query ?? value).trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue("");
    requestAnimationFrame(resizeTextarea);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="border-t border-border bg-bg px-4 py-4 sm:px-6">
      {showSamples && (
        <div className="mb-3 flex flex-wrap gap-2">
          {SAMPLE_QUERIES.map((query) => (
            <button
              key={query}
              type="button"
              onClick={() => handleSend(query)}
              className="rounded-full border border-border px-3 py-1.5 text-left text-xs text-text-secondary transition-colors hover:border-accent-dim hover:text-text-primary"
            >
              {query}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => {
            setValue(event.target.value);
            resizeTextarea();
          }}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
          rows={1}
          placeholder="Ask a question…"
          className="max-h-40 flex-1 resize-none rounded-xl border border-border bg-surface px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-accent-dim focus:outline-none disabled:opacity-50"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="shrink-0 rounded-xl border border-danger-dim bg-danger-dim px-4 py-2.5 text-sm font-medium text-text-primary transition-colors hover:bg-danger/20"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={() => handleSend()}
            disabled={!value.trim()}
            className="shrink-0 rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-bg transition-opacity disabled:opacity-40"
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}
