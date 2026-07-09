import { Fragment } from "react";

import { CitationChip } from "@/components/CitationChip";
import { tokenizeCitations } from "@/lib/citations";
import type { Source } from "@/lib/types";

interface CitationTextProps {
  text: string;
  sources: Source[];
}

/** Renders streamed answer text, turning `[n]` markers into citation chips. */
export function CitationText({ text, sources }: CitationTextProps) {
  const sourcesByNumber = new Map(sources.map((s) => [s.number, s]));

  return (
    <>
      {text.split("\n\n").map((paragraph, paragraphIndex, paragraphs) => (
        <p key={paragraphIndex} className={paragraphIndex < paragraphs.length - 1 ? "mb-3" : undefined}>
          {tokenizeCitations(paragraph).map((token, tokenIndex) => {
            if (token.kind === "text") {
              return <Fragment key={tokenIndex}>{token.value}</Fragment>;
            }
            return (
              <CitationChip
                key={tokenIndex}
                number={token.number}
                source={sourcesByNumber.get(token.number)}
              />
            );
          })}
        </p>
      ))}
    </>
  );
}
