"use client";

import { useState } from "react";

interface ErrorBubbleProps {
  message: string;
  detail?: string;
}

export function ErrorBubble({ message, detail }: ErrorBubbleProps) {
  const [showDetail, setShowDetail] = useState(false);

  return (
    <div className="rounded-xl border border-danger-dim bg-danger-dim px-4 py-3 text-sm text-text-primary">
      <div className="flex items-start gap-2">
        <span aria-hidden="true">⚠</span>
        <div className="flex-1">
          <p>{message}</p>
          {detail && (
            <>
              <button
                type="button"
                onClick={() => setShowDetail((v) => !v)}
                className="mt-1 text-xs text-text-secondary underline decoration-dotted hover:text-text-primary"
              >
                {showDetail ? "Hide details" : "Show details"}
              </button>
              {showDetail && (
                <pre className="mt-2 overflow-x-auto rounded-lg border border-border bg-bg p-2 text-xs text-text-secondary">
                  {detail}
                </pre>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
