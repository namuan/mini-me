#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "openai",
# ]
# ///
"""
mini-me.py - A TDD coding agent orchestrated by the app.
"""

import contextlib
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ─── Colors ────────────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    CYAN   = "\033[36m"

def header(msg: str)-> str: return f"{C.BOLD}{C.CYAN}{msg}{C.RESET}"
def step(msg: str)  -> str: return f"{C.CYAN}{msg}{C.RESET}"
def metric(msg: str)-> str: return f"{C.DIM}{msg}{C.RESET}"
def ok(msg: str)    -> str: return f"{C.GREEN}{msg}{C.RESET}"
def err(msg: str)   -> str: return f"{C.RED}{msg}{C.RESET}"
def cmd(msg: str)   -> str: return f"{C.DIM}{msg}{C.RESET}"

# ─── Configuration ─────────────────────────────────────────────────────────────

WORKSPACE = Path.cwd() / ".mini-me"
SESSIONS_DIR = WORKSPACE / "sessions"
MEMORY_DIR = WORKSPACE / "memory"

BASE_URL = "http://localhost:2276/v1"
DEFAULT_MODEL = "qwen3-4b"

SYSTEM_PROMPT = f"""\
You are a coding agent. Your only tool is bash — use it for everything.

## Reading files
Always read a file before editing it: cat, head, tail, grep, find, ls.

## Editing files
Use patch with a unified diff as your primary strategy:

  patch path/to/file << 'EOF'
  --- a/path/to/file
  +++ b/path/to/file
  @@ -10,6 +10,7 @@
   context line
  -old line
  +new line
   context line
  EOF

- Include 3 lines of unchanged context above and below each hunk.
- If patch rejects, re-read the file and regenerate the diff — never force-apply.
- For new files or full rewrites, use tee with a heredoc.

## Tests
Tests are first-class artifacts — they must outlive the session.
- Detect the language and test framework from existing project files before writing any test.
- Write every test to a real file in the project's conventional test directory
  (e.g. tests/test_<module>.py for pytest, __tests__/<module>.test.ts for Jest).
- Follow the framework's naming and structure conventions so the test is automatically
  discovered by the test runner with no extra configuration.
- Never write tests inline in a script or in a temporary file.

## Memory
Persistent memory lives in {MEMORY_DIR}.
- Save: tee {MEMORY_DIR}/<key>.md << 'EOF' ... EOF
- Load: cat {MEMORY_DIR}/<key>.md
- List: ls {MEMORY_DIR}/
Load relevant memory at the start of each session to recall prior context.
"""

# ─── Tool ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command. Use this for all file I/O, edits, and memory.",
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

# ─── Bash execution ────────────────────────────────────────────────────────────

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, executable="/bin/bash",
        )
        return (result.stdout + result.stderr) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Error: {e}"


def run_bash_with_code(command: str) -> tuple[str, int]:
    """Run a command and return (output, exit_code)."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, executable="/bin/bash",
        )
        return (result.stdout + result.stderr) or "(no output)", result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds.", 1
    except Exception as e:
        return f"Error: {e}", 1


# ─── Session management ────────────────────────────────────────────────────────

def session_path(session_key: str) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"{session_key.replace(':', '_')}.jsonl"


def load_session(session_key: str) -> list:
    path = session_path(session_key)
    if not path.exists():
        return []
    messages = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return messages


def append_message(session_key: str, message: dict) -> None:
    with session_path(session_key).open("a") as f:
        f.write(json.dumps(message) + "\n")


def save_session(session_key: str, messages: list) -> None:
    with session_path(session_key).open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


# ─── Context compaction ────────────────────────────────────────────────────────

def estimate_tokens(messages: list) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


def compact_session(client: OpenAI, session_key: str, messages: list) -> list:
    if estimate_tokens(messages) < 100_000:
        return messages
    split = len(messages) // 2
    old, recent = messages[:split], messages[split:]
    print("  📦 Compacting session history...")
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this coding session. Preserve files changed, decisions made, "
                "and tasks remaining:\n\n" + json.dumps(old, indent=2)
            ),
        }],
    )
    summary = response.choices[0].message.content or ""
    compacted = [{"role": "user", "content": f"[Session summary]\n{summary}"}] + recent
    save_session(session_key, compacted)
    return compacted


# ─── Agent turn ────────────────────────────────────────────────────────────────

def agent_turn(client: OpenAI, session_key: str, prompt: str) -> str:
    """
    Send a focused prompt to the agent. The agent may call bash multiple times.
    Returns the agent's final text response.
    """
    messages = load_session(session_key)
    messages = compact_session(client, session_key, messages)

    user_msg = {"role": "user", "content": prompt}
    messages.append(user_msg)
    append_message(session_key, user_msg)

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    while True:
        # Retry API call on transient errors
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    max_tokens=4096,
                    tools=TOOLS,
                    messages=[system_msg] + messages,
                )
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"  API error ({e}) — retrying in {wait}s")
                time.sleep(wait)

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
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                command = args.get("command", "")
                print(cmd(f"    $ {command}"))
                output = run_bash(command) if command else "(empty command)"
                print(cmd(f"    {output[:300].strip()}"))

                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": output}
                messages.append(tool_msg)
                append_message(session_key, tool_msg)


# ─── Timing ───────────────────────────────────────────────────────────────────

_timings: list[tuple[str, float]] = []


@contextlib.contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _timings.append((label, elapsed))
    print(metric(f"  ⏱  {label}: {elapsed:.1f}s"))


def print_summary() -> None:
    if not _timings:
        return
    total = sum(t for _, t in _timings)
    print(metric("\n── Timing summary"))
    for label, elapsed in _timings:
        pct = elapsed / total * 100
        print(metric(f"  {elapsed:6.1f}s  {pct:4.0f}%  {label}"))
    print(metric(f"  {'─'*30}"))
    print(metric(f"  {total:6.1f}s  total"))


# ─── TDD orchestration ─────────────────────────────────────────────────────────

def parse_tasks(response: str) -> list[str]:
    """Extract a numbered list of tasks from the agent's planning response."""
    tasks = []
    for line in response.splitlines():
        m = re.match(r"^\s*\d+[.)]\s+(.+)", line)
        if m:
            tasks.append(m.group(1).strip())
    return tasks


def extract_test_command(response: str) -> str:
    """Parse TEST_COMMAND: <cmd> from the agent's test-writing response."""
    m = re.search(r"TEST_COMMAND:\s*(.+)", response)
    if not m:
        return ""
    # Normalize all Unicode whitespace variants (non-breaking spaces, thin spaces, etc.)
    # to plain ASCII spaces so bash tokenises the command correctly.
    return " ".join(m.group(1).strip().split())


def update_plan(tasks: list[str], current: int, status: str) -> None:
    """Write current plan state to memory/plan.md."""
    lines = [f"# Plan\n"]
    for i, task in enumerate(tasks):
        if i < current:
            lines.append(f"- [x] {task}")
        elif i == current:
            lines.append(f"- [{status}] {task}")
        else:
            lines.append(f"- [ ] {task}")
    (MEMORY_DIR / "plan.md").write_text("\n".join(lines) + "\n")


def tdd_cycle(client: OpenAI, session_key: str, requirement: str) -> None:
    _timings.clear()

    # ── Planning ──────────────────────────────────────────────────────────────
    print(header("\n── Planning"))
    with timed("planning"):
        plan_response = agent_turn(
            client, session_key,
            f"First, inspect the project to determine:\n"
            f"  1. The implementation language\n"
            f"  2. The test framework and its test discovery conventions (file names, directories)\n"
            f"  3. The command to run the full test suite\n"
            f"Save this to {MEMORY_DIR}/project.md, then decompose the requirement below "
            f"into atomic tasks, each small enough to be implemented and tested in isolation.\n\n"
            f"Output a numbered list only — one task per line, no other text.\n\n"
            f"Requirement: {requirement}"
        )
    tasks = parse_tasks(plan_response)
    if not tasks:
        print("  Agent did not return a parseable task list.")
        print(plan_response)
        return

    update_plan(tasks, -1, " ")
    print(step(f"  {len(tasks)} tasks planned"))
    for i, t in enumerate(tasks, 1):
        print(step(f"    {i}. {t}"))

    # ── Per-task TDD loop ─────────────────────────────────────────────────────
    for i, task in enumerate(tasks):
        print(header(f"\n── Task {i + 1}/{len(tasks)}: {task}"))
        update_plan(tasks, i, "~")

        # ── Step 1: Write test ────────────────────────────────────────────────
        print(step("  Step 1: Write test"))
        test_cmd = ""
        attempt = 0
        while True:
            attempt += 1
            with timed(f"task {i+1} · write test (attempt {attempt})"):
                response = agent_turn(
                    client, session_key,
                    f"Task: {task}\n\n"
                    f"Write a failing test for this task and nothing else — no implementation code.\n"
                    f"The test must be written to the correct test file on disk following the "
                    f"project's language and framework conventions (check {MEMORY_DIR}/project.md).\n"
                    f"The test file must be discoverable and runnable by the project's test runner "
                    f"without any extra configuration.\n"
                    f"End your response with exactly:\n"
                    f"TEST_COMMAND: <the bash command to run only this test>"
                )
            test_cmd = extract_test_command(response)
            if not test_cmd:
                print(err("  No TEST_COMMAND found — asking agent to clarify"))
                with timed(f"task {i+1} · clarify TEST_COMMAND"):
                    response = agent_turn(
                        client, session_key,
                        "Your response did not include a TEST_COMMAND line. "
                        "Reply with only the exact bash command to run the test, "
                        "on its own line prefixed with TEST_COMMAND:"
                    )
                test_cmd = extract_test_command(response)
            if not test_cmd:
                print(err("  Still no TEST_COMMAND — retrying from scratch"))
                continue

            print(step(f"  Running: {test_cmd}"))
            with timed(f"task {i+1} · run test (red check)"):
                output, code = run_bash_with_code(test_cmd)
            print(metric(f"  Exit code: {code}"))
            print(f"  {output[:400].strip()}")

            if code != 0:
                print(ok("  ✓ Test is RED — correct"))
                break
            else:
                print(err("  ✗ Test passes before implementation — test is invalid, asking agent to fix"))
                with timed(f"task {i+1} · fix invalid test"):
                    agent_turn(
                        client, session_key,
                        f"The test passed before any implementation exists (exit 0):\n\n{output}\n\n"
                        f"A test that passes before implementation is not a real test. "
                        f"Fix the test so it fails for the correct reason, then end with TEST_COMMAND: <cmd>"
                    )

        # Save progress after red test confirmed
        update_plan(tasks, i, "T")
        run_bash(f"tee {MEMORY_DIR}/progress.md << 'EOF'\n"
                 f"task: {task}\nstatus: test-written\ntest_command: {test_cmd}\nEOF")

        # ── Step 2: Write implementation ──────────────────────────────────────
        print(step("  Step 2: Implement"))
        failure_output = output
        impl_attempt = 0
        while True:
            impl_attempt += 1
            with timed(f"task {i+1} · implement (attempt {impl_attempt})"):
                agent_turn(
                    client, session_key,
                    f"The test is failing:\n\n{failure_output}\n\n"
                    f"Write the minimum implementation to make this pass:\n"
                    f"  {test_cmd}\n\n"
                    f"Do not modify the test file."
                )
            print(step(f"  Running: {test_cmd}"))
            with timed(f"task {i+1} · run test (green check)"):
                output, code = run_bash_with_code(test_cmd)
            print(metric(f"  Exit code: {code}"))
            print(f"  {output[:400].strip()}")

            if code == 0:
                print(ok("  ✓ Test is GREEN"))
                break
            else:
                print(err("  ✗ Still failing — retrying implementation"))
                failure_output = output

        # Save progress and accumulate test command
        update_plan(tasks, i, "x")
        run_bash(f"tee {MEMORY_DIR}/progress.md << 'EOF'\n"
                 f"task: {task}\nstatus: complete\ntest_command: {test_cmd}\nEOF")
        run_bash(f"echo {test_cmd!r} >> {MEMORY_DIR}/test_commands.txt")

    update_plan(tasks, len(tasks), " ")

    # ── Regression run ────────────────────────────────────────────────────────
    print(header("\n── Regression check"))
    with timed("regression suite"):
        regression_response = agent_turn(
            client, session_key,
            f"All tasks are complete. Read {MEMORY_DIR}/project.md and run the full test suite "
            f"to confirm there are no regressions. Report which tests passed and which failed."
        )
    print(f"  {regression_response.strip()}")

    print_summary()
    print(ok("\n── All tasks complete"))


# ─── REPL ─────────────────────────────────────────────────────────────────────

def main():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
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

        try:
            tdd_cycle(client, session_key, user_input)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        except Exception as e:
            print(f"\n  Error: {e}")


if __name__ == "__main__":
    main()
