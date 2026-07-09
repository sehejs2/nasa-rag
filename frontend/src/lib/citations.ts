/**
 * Splits answer text into plain-text and citation-marker tokens so the UI can
 * render `[n]` as an inline chip. Not part of the SSE protocol itself - pure
 * text-rendering logic, kept separate from chatStream.ts.
 */

export type TextToken =
  | { kind: "text"; value: string }
  | { kind: "citation"; number: number };

const CITATION_PATTERN = /\[(\d+)\]/g;

export function tokenizeCitations(text: string): TextToken[] {
  const tokens: TextToken[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(CITATION_PATTERN)) {
    const index = match.index;
    if (index > lastIndex) {
      tokens.push({ kind: "text", value: text.slice(lastIndex, index) });
    }
    tokens.push({ kind: "citation", number: Number(match[1]) });
    lastIndex = index + match[0].length;
  }

  if (lastIndex < text.length) {
    tokens.push({ kind: "text", value: text.slice(lastIndex) });
  }

  return tokens;
}
