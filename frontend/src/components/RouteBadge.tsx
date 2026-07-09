import type { Route } from "@/lib/types";

interface RouteBadgeProps {
  route: Route;
  tools: string[];
}

function label(route: Route, tools: string[]): { icon: string; text: string } {
  switch (route) {
    case "retrieval":
      return { icon: "📄", text: "Retrieved docs" };
    case "tools":
      return { icon: "🛰", text: tools.length > 0 ? `Live tool: ${tools.join(", ")}` : "Live tool" };
    case "both":
      return { icon: "📄+🛰", text: "Docs + live tools" };
    case "direct":
      return { icon: "💬", text: "Direct" };
  }
}

export function RouteBadge({ route, tools }: RouteBadgeProps) {
  const { icon, text } = label(route, tools);

  return (
    <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-accent-dim bg-accent-dim px-2.5 py-1 text-xs font-medium tracking-wide text-accent-strong">
      <span aria-hidden="true">{icon}</span>
      <span>{text}</span>
    </span>
  );
}
