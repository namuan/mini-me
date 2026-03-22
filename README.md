# mini-me

A minimal multi-agent AI assistant with persistent sessions, long-term memory, and tool use.

## Usage

```bash
./mini-me.py
```

Requires [uv](https://docs.astral.sh/uv/). Dependencies are declared inline — no manual install needed.

## Agents

- **Jarvis** (default) — general-purpose assistant
- **Scout** — research specialist, invoke with `/research <query>`

## Commands

| Command | Description |
|---|---|
| `/research <query>` | Route to the Scout research agent |
| `/new` | Start a fresh session |
| `/quit` | Exit |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible API endpoint |
| `DEFAULT_MODEL` | `local-model` | Model name passed to the API |

## Workspace

Sessions and memory are stored in `.mini-openclaw/` in the current directory.
