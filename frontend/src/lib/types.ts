/**
 * Types mirroring the backend SSE event protocol exactly.
 * See CLAUDE.md "SSE event protocol" (app/agent/chat_stream.py) - this is a
 * pure consumer of that contract, not a place to invent new fields.
 */

export type Route = "retrieval" | "tools" | "both" | "direct";

export interface MetaEventData {
  route: Route;
  tools: string[];
  iterations: number;
}

export type SourceKind = "chunk" | "tool";

export interface Source {
  number: number;
  kind: SourceKind;
  title: string;
  url: string | null;
  detail: string;
  ref_id: string;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface DoneEventData {
  total_latency_ms: number;
  token_usage: TokenUsage;
  cited_sources: number[];
  invalid_citations: number[];
}

export interface ErrorEventData {
  message: string;
}

export type ChatEvent =
  | { type: "meta"; data: MetaEventData }
  | { type: "sources"; data: Source[] }
  | { type: "delta"; data: { text: string } }
  | { type: "done"; data: DoneEventData }
  | { type: "error"; data: ErrorEventData };

/**
 * Client-side conversation state, not part of the backend protocol. Each
 * assistant turn accumulates the events above as they stream in.
 */
export interface UserMessage {
  role: "user";
  id: string;
  text: string;
}

export interface AssistantMessage {
  role: "assistant";
  id: string;
  status: "streaming" | "done" | "error";
  stopped?: boolean;
  meta?: MetaEventData;
  sources?: Source[];
  text: string;
  doneData?: DoneEventData;
  errorMessage?: string;
  errorDetail?: string;
}

export type ConversationMessage = UserMessage | AssistantMessage;

