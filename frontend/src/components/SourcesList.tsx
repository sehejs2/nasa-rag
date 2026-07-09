import type { Source } from "@/lib/types";

interface SourcesListProps {
  sources: Source[];
}

export function SourcesList({ sources }: SourcesListProps) {
  if (sources.length === 0) return null;

  return (
    <details className="mt-3 text-sm text-text-secondary">
      <summary className="cursor-pointer select-none font-medium text-text-secondary hover:text-text-primary">
        Sources ({sources.length})
      </summary>
      <ol className="mt-2 flex flex-col gap-2 border-l border-border pl-3">
        {sources.map((source) => (
          <li key={source.number} className="text-xs leading-relaxed">
            <span className="mr-1.5 font-mono text-text-muted">[{source.number}]</span>
            <span className="font-medium text-text-primary">{source.title}</span>
            <span className="text-text-muted"> — {source.detail}</span>
            {source.url && (
              <>
                {" "}
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent hover:text-accent-strong hover:underline"
                >
                  ↗
                </a>
              </>
            )}
          </li>
        ))}
      </ol>
    </details>
  );
}
