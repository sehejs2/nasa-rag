"""Manual invocation CLI for POST /chat. Usage: make chat q="your question".

Also serves as a reference SSE consumer for the Phase 8 frontend: it drives
the app in-process over an ASGI transport (no separate `make dev` needed) and
parses the exact same event protocol a real browser client would.
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

from app.main import app


def _parse_sse_block(block: str) -> tuple[str | None, str]:
    event_name = None
    data_lines = []
    for line in block.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    return event_name, "\n".join(data_lines)


async def _run(query: str) -> None:
    sources_payload = None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60.0) as client:
        async with client.stream("POST", "/chat", json={"query": query}) as response:
            response.raise_for_status()
            buffer = ""
            async for raw_chunk in response.aiter_text():
                buffer += raw_chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    event_name, data = _parse_sse_block(block)
                    if not event_name:
                        continue
                    payload = json.loads(data) if data else None

                    if event_name == "meta":
                        print(
                            f"Route: {payload['route']}  tools: {payload['tools']}  "
                            f"iterations: {payload['iterations']}"
                        )
                        print()
                    elif event_name == "sources":
                        sources_payload = payload
                    elif event_name == "delta":
                        print(payload["text"], end="", flush=True)
                    elif event_name == "done":
                        print("\n")
                        print(
                            f"[done] latency={payload['total_latency_ms']:.0f}ms "
                            f"tokens={payload['token_usage']['total_tokens']}"
                        )
                        print(
                            f"       cited_sources={payload['cited_sources']} "
                            f"invalid_citations={payload['invalid_citations']}"
                        )
                    elif event_name == "error":
                        print(f"\n[error] {payload['message']}", file=sys.stderr)

    if sources_payload:
        print()
        print("Sources:")
        for source in sources_payload:
            url_part = f" ({source['url']})" if source.get("url") else ""
            print(f"  [{source['number']}] {source['title']}{url_part} - {source['detail']}")


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: make chat q="your question"', file=sys.stderr)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1]))


if __name__ == "__main__":
    main()
