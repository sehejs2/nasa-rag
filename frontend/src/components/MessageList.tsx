import { AssistantMessageView } from "@/components/AssistantMessageView";
import { UserMessageView } from "@/components/UserMessageView";
import type { ConversationMessage } from "@/lib/types";

export function MessageList({ messages }: { messages: ConversationMessage[] }) {
  return (
    <div className="flex flex-1 flex-col gap-5 overflow-y-auto px-4 py-6 sm:px-6">
      {messages.map((message) =>
        message.role === "user" ? (
          <UserMessageView key={message.id} message={message} />
        ) : (
          <AssistantMessageView key={message.id} message={message} />
        ),
      )}
    </div>
  );
}
