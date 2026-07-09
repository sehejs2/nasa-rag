"use client";

import { useEffect, useRef, useState } from "react";

import type { Source } from "@/lib/types";

interface CitationChipProps {
  number: number;
  source: Source | undefined;
}

/**
 * Renders `[n]` as a small superscript chip. Valid citations (a matching
 * source) get an interactive popover on hover (desktop) or tap (mobile).
 * Invalid numbers (no matching source - the composer cited something that
 * doesn't exist) render as plain muted text, never a broken chip.
 */
export function CitationChip({ number, source }: CitationChipProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  if (!source) {
    return <sup className="mx-0.5 text-[0.85em] text-text-muted">[{number}]</sup>;
  }

  return (
    <span
      ref={containerRef}
      className="relative inline-block"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <sup>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="mx-0.5 rounded border border-accent-dim bg-accent-dim px-1.5 py-px text-[0.85em] font-bold text-accent-strong transition-colors hover:bg-accent/20"
        >
          {number}
        </button>
      </sup>

      {open && (
        <span
          role="tooltip"
          className="popover-enter absolute bottom-full left-1/2 z-20 mb-2 w-64 -translate-x-1/2 rounded-lg border border-border-strong bg-surface-2 p-3 text-left text-xs normal-case text-text-secondary shadow-lg shadow-black/40"
        >
          <span className="mb-1 flex items-center gap-1.5">
            <span className="rounded border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-text-muted">
              {source.kind === "chunk" ? "Document" : "Tool call"}
            </span>
          </span>
          <span className="block font-medium text-text-primary">{source.title}</span>
          <span className="mt-1 block text-text-secondary">{source.detail}</span>
          {source.url && (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-2 inline-block text-accent hover:text-accent-strong hover:underline"
            >
              Open source ↗
            </a>
          )}
        </span>
      )}
    </span>
  );
}
