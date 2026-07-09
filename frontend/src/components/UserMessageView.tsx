import type { UserMessage } from "@/lib/types";

export function UserMessageView({ message }: { message: UserMessage }) {
  return (
    <div className="max-w-[42rem] self-end rounded-2xl rounded-tr-sm border border-border-strong bg-surface-2 px-4 py-3 text-[0.95rem] leading-relaxed text-text-primary">
      {message.text}
    </div>
  );
}
