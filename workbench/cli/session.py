"""The asyncio plumbing: one task reads the socket, one reads the keyboard.

Deliberately thin -- all protocol decisions live in client.py. The only real
design point here is that receiving must not block on typing, or a streaming
reply would not appear until the user pressed a key."""
import asyncio
import json
import sys

from workbench.cli.client import (HELP, QUIT, RenderState, completes,
                                  parse_input, render_event)

DEFAULT_URL = "ws://127.0.0.1:8321/ws"


async def _receive(ws, state: RenderState, done: "asyncio.Event",
                   waiting: dict) -> None:
    async for raw in ws:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") in waiting["for"]:
            done.set()
        out = render_event(event, state)
        if out is None:
            continue
        # Tokens stream inline; everything else is a framed line. Without the
        # leading newline a status line would land mid-word in a live reply.
        if event.get("type") == "token":
            sys.stdout.write(out)
        else:
            sys.stdout.write(("\n" if state.streaming else "") + out + "\n")
        sys.stdout.flush()


async def _read_line(prompt: str) -> str:
    """Read stdin without blocking the event loop, so tokens keep printing."""
    return await asyncio.get_running_loop().run_in_executor(None, input, prompt)


async def run(url: str = DEFAULT_URL) -> int:
    import websockets

    try:
        ws = await websockets.connect(url, max_size=None)
    except OSError as e:
        print(f"cannot reach the server at {url}: {e}", file=sys.stderr)
        print("start it with:  uv run python -m workbench.server.app", file=sys.stderr)
        return 1

    state = RenderState()
    # Piped stdin outruns the server: without this, every line is read and
    # /quit closes the socket before the first token lands. Interactive use
    # keeps the old behaviour, where typing is what paces the session.
    batch = not sys.stdin.isatty()
    completed, waiting = asyncio.Event(), {"for": set()}
    if not batch:
        print(f"connected to {url}.  /help for commands, /quit to exit.")
    receiver = asyncio.create_task(_receive(ws, state, completed, waiting))
    try:
        while True:
            try:
                line = await _read_line("" if (state.streaming or batch)
                                        else "\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if receiver.done():
                print("connection closed by the server.")
                break

            if line.strip() == "/help":
                print(HELP)
                continue
            if line.strip() == "/logprobs":
                state.show_logprobs = not state.show_logprobs
                print(f"[logprobs {'on' if state.show_logprobs else 'off'}]")
                continue

            msg = parse_input(line, state)
            if msg is QUIT:
                break
            if msg is None:
                if line.strip():
                    print("[unrecognised -- /help for commands]")
                continue
            if msg["type"] == "preview_edit":
                state.pending_edit = msg["event"]
            if msg["type"] == "user_message":
                state.output_tokens = 0
                if batch:
                    print(f"> {line.strip()}")
            expect = completes(msg["type"])
            waiting["for"] = expect if batch else set()
            completed.clear()
            await ws.send(json.dumps(msg))
            if batch and expect:
                # One turn at a time. Bounded so a server that never replies
                # fails the script instead of hanging it.
                try:
                    await asyncio.wait_for(completed.wait(), timeout=600)
                except asyncio.TimeoutError:
                    print("[timed out waiting for the server]", file=sys.stderr)
                    break
    finally:
        receiver.cancel()
        await ws.close()
    return 0


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    try:
        return asyncio.run(run(url))
    except KeyboardInterrupt:
        return 130
