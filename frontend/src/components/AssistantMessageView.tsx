import { CitationText } from "@/components/CitationText";
import { ErrorBubble } from "@/components/ErrorBubble";
import { RouteBadge } from "@/components/RouteBadge";
import { SourcesList } from "@/components/SourcesList";
import type { AssistantMessage } from "@/lib/types";

function formatLatency(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export function AssistantMessageView({ message }: { message: AssistantMessage }) {
  const sources = message.sources ?? [];
  const isStreaming = message.status === "streaming";

  return (
    <div className="flex max-w-[42rem] flex-col items-start gap-2">
      {message.meta && <RouteBadge route={message.meta.route} tools={message.meta.tools} />}

      {message.status === "error" ? (
        <ErrorBubble message={message.errorMessage ?? "Something went wrong."} detail={message.errorDetail} />
      ) : (
        <div className="rounded-2xl rounded-tl-sm border border-border bg-surface px-4 py-3 text-[0.95rem] leading-relaxed text-text-primary">
          {message.text.length > 0 ? (
            <CitationText text={message.text} sources={sources} />
          ) : isStreaming ? (
            <span className="text-text-muted">Thinking…</span>
          ) : null}
          {isStreaming && <span className="streaming-cursor" aria-hidden="true" />}
        </div>
      )}

      {message.status === "done" && (
        <div className="flex w-full flex-col gap-1">
          <SourcesList sources={sources} />
          <p className="text-xs text-text-muted">
            {message.stopped ? "Stopped" : message.doneData ? formatLatency(message.doneData.total_latency_ms) : null}
          </p>
        </div>
      )}
    </div>
  );
}
