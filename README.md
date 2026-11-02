# qwen-local-mcp

An MCP server that lets Codex, Claude Code, or any MCP client delegate a bounded
coding task to a **local** Qwen model running in [Ollama](https://ollama.com).

The big model plans; the small local model does the grunt work (read, search,
edit, run tests) inside one repository, with hard guardrails:

- Edits are confined to a workspace under a configured repositories root.
- No deletes, renames, commits, pushes, package installs, or network access.
- Refuses to edit on `main`/`master`; use a feature branch or worktree.
- Only allowlisted commands run (pytest, unittest, ruff, mypy, npm/pnpm/yarn
  test/lint/build/typecheck, dotnet test/build, cargo, go, cmake/ctest, and
  read-only `git status`/`git diff`). No shell is ever spawned.
- Secret files (`.env`, keys, `credentials.json`, ...) and dependency/build
  directories are invisible to the model.
- Tool results are clipped so a 32k context never silently loses the system prompt.
- Stops early when the model repeats a rejected call, and keeps a wall-clock
  budget under Codex's tool timeout so you always get a report back.

## Requirements

- Windows, macOS, or Linux with Python 3.11+
- [Ollama](https://ollama.com) running locally with the model pulled:

```bash
ollama pull qwen3.5:9b
```

## Install

```bash
git clone https://github.com/d1dac0/qwen-local-mcp.git
cd qwen-local-mcp
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS / Linux
```

### Claude Code

```bash
claude mcp add --scope user --transport stdio qwen-local -- "<path>/.venv/Scripts/python.exe" "<path>/server.py"
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.qwen_local]
command = "<path>/.venv/Scripts/python.exe"
args = ["<path>/server.py"]
cwd = "<path>"
enabled = true
enabled_tools = ["qwen_delegate"]
startup_timeout_sec = 120
tool_timeout_sec = 900
```

## Usage

The server exposes one tool, `qwen_delegate(task, workspace)`:

- `workspace`: absolute path of the repository (or a subfolder) to work in.
  Must be under the repositories root.
- `task`: a self-contained instruction naming the files and the acceptance
  check, e.g. *"test_calc.py fails. Fix the bug in calc.py so `python -m pytest -q`
  passes, then run it and report the exit code."*

The result is a report with the model's summary, every tool call, and the
exact commands and exit codes it ran.

## Configuration

All optional, via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `QWEN_WORKER_REPOSITORIES_ROOT` | `~/source/repos` | Every workspace must live under this root |
| `QWEN_WORKER_MODEL` | `qwen3.5:9b` | Ollama model tag |
| `QWEN_WORKER_CONTEXT_TOKENS` | `32768` | Ollama `num_ctx` |
| `QWEN_WORKER_MAX_STEPS` | `20` | Max agent turns per task |
| `QWEN_WORKER_MAX_SECONDS` | `840` | Wall-clock budget per task |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama base URL |

Python tooling (`python`, `pytest`, `ruff`, `mypy`) is resolved from the
workspace's own `.venv`/`venv` when present, so tests see the project's dependencies.

## Tools the model gets

| Tool | Notes |
|---|---|
| `list_files` | Skips VCS, dependency, cache, and build-output dirs |
| `read_file` | Paged with `start_line`/`max_lines`; UTF-8 only |
| `search_code` | Case-insensitive literal search, `path:line: text` |
| `edit_file` | One exact, unique replacement; preserves CRLF/LF |
| `create_file` | Never overwrites |
| `run_command` | Allowlisted executables, argument paths checked, no shell |

## License

MIT
