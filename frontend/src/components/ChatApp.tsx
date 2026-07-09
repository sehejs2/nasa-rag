"use client";

import { useEffect, useRef, useState } from "react";

import { ChatInputBar } from "@/components/ChatInputBar";
import { MessageList } from "@/components/MessageList";
import { API_URL } from "@/lib/config";
import { streamChat } from "@/lib/chatStream";
import type { AssistantMessage, ConversationMessage } from "@/lib/types";

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function ChatApp() {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    // Cancel any in-flight request if the whole app unmounts.
    return () => abortControllerRef.current?.abort();
  }, []);

  function patchAssistantMessage(id: string, patch: Partial<AssistantMessage>) {
    setMessages((prev) =>
      prev.map((message) => (message.role === "assistant" && message.id === id ? { ...message, ...patch } : message)),
    );
  }

  function appendAssistantText(id: string, text: string) {
    setMessages((prev) =>
      prev.map((message) =>
        message.role === "assistant" && message.id === id ? { ...message, text: message.text + text } : message,
      ),
    );
  }

  function sendQuery(query: string) {
    if (isStreaming) return;

    const userMessage: ConversationMessage = { role: "user", id: newId(), text: query };
    const assistantId = newId();
    const assistantMessage: AssistantMessage = { role: "assistant", id: assistantId, status: "streaming", text: "" };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    (async () => {
      try {
        for await (const event of streamChat(query, { apiUrl: API_URL, signal: controller.signal })) {
          switch (event.type) {
            case "meta":
              patchAssistantMessage(assistantId, { meta: event.data });
              break;
            case "sources":
              patchAssistantMessage(assistantId, { sources: event.data });
              break;
            case "delta":
              appendAssistantText(assistantId, event.data.text);
              break;
            case "done":
              patchAssistantMessage(assistantId, { status: "done", doneData: event.data });
              break;
            case "error":
              patchAssistantMessage(assistantId, {
                status: "error",
                errorMessage: "The assistant hit an error while answering.",
                errorDetail: event.data.message,
              });
              break;
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          patchAssistantMessage(assistantId, { status: "done", stopped: true });
        } else {
          const detail = err instanceof Error ? err.message : String(err);
          patchAssistantMessage(assistantId, {
            status: "error",
            errorMessage: `Couldn't reach the API at ${API_URL}. Check that the backend is running (make dev).`,
            errorDetail: detail,
          });
        }
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    })();
  }

  function stopStreaming() {
    abortControllerRef.current?.abort();
  }

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-border px-4 py-4 sm:px-6">
        <h1 className="text-sm font-semibold tracking-wide text-text-primary">NASA RAG</h1>
        <p className="text-xs text-text-muted">Agentic retrieval over NASA mission reports + live NASA data</p>
      </header>

      {messages.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center">
          <p className="text-lg font-medium text-text-secondary">Ask about NASA missions or live data.</p>
          <p className="max-w-sm text-sm text-text-muted">
            Try a sample question below, or ask your own. Every question starts a fresh conversation.
          </p>
        </div>
      ) : (
        <>
          <MessageList messages={messages} />
          <div ref={scrollAnchorRef} />
        </>
      )}

      <ChatInputBar
        onSend={sendQuery}
        onStop={stopStreaming}
        isStreaming={isStreaming}
        showSamples={messages.length === 0}
      />
    </div>
  );
}
