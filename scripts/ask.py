"""Manual invocation CLI for the agent loop. Usage: make ask q="your question"."""

from __future__ import annotations

import asyncio
import sys

from app.agent.loop import run_agent
from app.agent.models import AgentTrace


def _print_trace(trace: AgentTrace) -> None:
    print(f"Route: {trace.route}  (stopped: {trace.stopped_reason}, iterations: {trace.iterations_used})")
    print(f"Total latency: {trace.total_latency_ms:.0f}ms  |  tokens: {trace.token_usage.total_tokens}")
    print()

    if trace.tool_calls:
        print("Tool calls:")
        for call in trace.tool_calls:
            status = "ok" if call.result_ok else "FAILED"
            print(f"  [{call.iteration}] {call.tool_name}({call.arguments}) -> {status} ({call.latency_ms:.0f}ms)")
            print(f"      {call.result_summary}")
    else:
        print("Tool calls: none")
    print()

    print("Draft answer:")
    print(trace.draft_answer)


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: make ask q="your question"', file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]
    trace = asyncio.run(run_agent(query))
    _print_trace(trace)


if __name__ == "__main__":
    main()
