#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "openai",
# ]
# ///
"""
mini-me.py - A coding agent with a single bash tool.
"""

import json
import os
import subprocess
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ─── Configuration ─────────────────────────────────────────────────────────────

WORKSPACE = Path.cwd() / ".mini-me"
SESSIONS_DIR = WORKSPACE / "sessions"
MEMORY_DIR = WORKSPACE / "memory"

BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "local-model"

SYSTEM_PROMPT = f"""\
You are a coding agent. Your only tool is bash — use it for everything.

## Reading files
Use cat, head, tail, grep, find, ls.
Always read a file before editing it.

## Editing files
Use patch with a unified diff as your default editing strategy:

```
patch path/to/file.py << 'EOF'
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,6 +10,7 @@
 context line
-old line
+new line
 context line
EOF
```

Rules:
- Include 3 lines of unchanged context above and below each hunk so the patch anchors correctly.
- If patch rejects, re-read the file and regenerate the diff — never force-apply.
- For new files or complete rewrites, use tee with a heredoc instead:
  tee path/to/file.py << 'EOF'
  # full content
  EOF

## Memory
Your persistent memory lives in {MEMORY_DIR}.
- Save: echo "..." | tee {MEMORY_DIR}/<key>.md
- Load: cat {MEMORY_DIR}/<key>.md
- List: ls {MEMORY_DIR}/
At the start of each conversation, load relevant memory files to recall context.

## Working style
- Run tests or linters after changes when available.
- Prefer small, focused patches over large rewrites.
- When stuck, inspect the actual error — don't guess.
"""

# ─── Tool ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command. Use this for all file I/O, code edits, tests, and memory operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"},
                },
                "required": ["command"],
            },
        },
    }
]

# ─── Safe Commands ─────────────────────────────────────────────────────────────

SAFE_PREFIXES = {
    "cat", "head", "tail", "grep", "find", "ls", "echo", "pwd",
    "which", "wc", "stat", "file", "diff", "git", "python", "python3",
    "node", "npm", "uv", "rg", "tree",
}


def needs_approval(command: str) -> bool:
    base = command.strip().split()[0].split("/")[-1]
    return base not in SAFE_PREFIXES


# ─── Tool Execution ───────────────────────────────────────────────────────────


def run_bash(command: str) -> str:
    if needs_approval(command):
        print(f"\n  ⚠️  {command}")
        answer = input("  Run? (y/n): ").strip().lower()
        if answer != "y":
            return "Cancelled by user."
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60,
            executable="/bin/bash",
        )
        output = result.stdout + result.stderr
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Error: {e}"


# ─── Session Management ───────────────────────────────────────────────────────


def get_session_path(session_key: str) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = session_key.replace(":", "_").replace("/", "_")
    return SESSIONS_DIR / f"{safe_key}.jsonl"


def load_session(session_key: str) -> list:
    path = get_session_path(session_key)
    messages = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return messages


def append_message(session_key: str, message: dict) -> None:
    with get_session_path(session_key).open("a") as f:
        f.write(json.dumps(message) + "\n")


def save_session(session_key: str, messages: list) -> None:
    with get_session_path(session_key).open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


# ─── Context Compaction ───────────────────────────────────────────────────────


def estimate_tokens(messages: list) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


def compact_session(session_key: str, messages: list) -> list:
    if estimate_tokens(messages) < 100_000:
        return messages
    split = len(messages) // 2
    old, recent = messages[:split], messages[split:]
    print("\n  📦 Compacting session history...")
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this coding session concisely. Preserve key decisions, "
                "files changed, and open tasks:\n\n"
                f"{json.dumps(old, indent=2)}"
            ),
        }],
    )
    summary = response.choices[0].message.content or ""
    compacted = [{"role": "user", "content": f"[Session summary]\n{summary}"}] + recent
    save_session(session_key, compacted)
    return compacted


# ─── Agent Loop ───────────────────────────────────────────────────────────────

session_locks: dict = defaultdict(threading.Lock)


def run_agent_turn(session_key: str, user_text: str) -> str:
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    with session_locks[session_key]:
        messages = load_session(session_key)
        messages = compact_session(session_key, messages)

        user_msg = {"role": "user", "content": user_text}
        messages.append(user_msg)
        append_message(session_key, user_msg)

        for _ in range(20):
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                max_tokens=4096,
                tools=TOOLS,
                messages=[system_msg] + messages,
            )

            choice = response.choices[0]
            msg = choice.message

            assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]

            messages.append(assistant_msg)
            append_message(session_key, assistant_msg)

            if choice.finish_reason == "stop":
                return msg.content or ""

            if choice.finish_reason == "tool_calls":
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    command = args.get("command", "")
                    print(f"  $ {command}")
                    result = run_bash(command)
                    print(f"  {result[:200].strip()}")

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                    messages.append(tool_msg)
                    append_message(session_key, tool_msg)

        return "(max turns reached)"


# ─── REPL ─────────────────────────────────────────────────────────────────────


def main():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    session_key = "coding:repl"
    print("mini-me coding agent")
    print(f"  Model:     {DEFAULT_MODEL}")
    print(f"  Endpoint:  {BASE_URL}")
    print(f"  Workspace: {WORKSPACE}")
    print("  Commands:  /new  /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("Bye!")
            break
        if user_input.lower() == "/new":
            session_key = f"coding:repl:{datetime.now().strftime('%Y%m%d%H%M%S')}"
            print("  Session reset.\n")
            continue

        response = run_agent_turn(session_key, user_input)
        print(f"\n🤖 {response}\n")


if __name__ == "__main__":
    main()
