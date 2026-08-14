# Install

## With uv (recommended)

```bash
# try it without installing anything
uvx --from "membukkit[all]" membukkit ui --demo personal-assistant

# or put the CLI on your PATH, isolated, no venv to manage
uv tool install "membukkit[all]"
membukkit ui --demo personal-assistant
```

## With pip

```bash
pip install "membukkit[all]"
membukkit ui --demo personal-assistant
```

## From git

To run unreleased `main`, or to install a fork:

```bash
pip install "membukkit[all] @ git+https://github.com/memseekai/membukkit.git"
```

## With Docker

```bash
docker run -p 127.0.0.1:8377:8377 -v membukkit-data:/data \
  -e OPENAI_API_KEY ghcr.io/memseekai/membukkit
# then open http://localhost:8377
```

Volumes, environment variables, Compose, and running the multi-tenant service in a
container: **[Docker](docker.md)**.

## Requirements

- **Python 3.10+.** Node is not needed: the GUI ships prebuilt inside the package.
- **An LLM.** Paste a key in the GUI (**keys** in the sidebar, saved to
  `~/.membukkit/credentials.env`), `export OPENAI_API_KEY=sk-…`, or stay local with
  `--llm ollama:llama3.1`.
- **Network on first run**, to fetch the retrieval weights. Everything after that is local,
  and stores live under `~/.membukkit/` (override with `MEMBUKKIT_HOME`).

## Extras

`[all]` is the right default. The rest matter when you want a smaller install.

| Extra | What it unlocks |
|---|---|
| *(none)* | Library + CLI. The GUI is **not** included. |
| `service` | `membukkit ui` (local GUI) and `membukkit serve` (multi-tenant service) |
| `mcp` | MCP stdio server (`membukkit mcp`) for Cursor / Claude Desktop |
| `bm25` | Optional BM25 lexical lane ([off by default](library.md#lexical-lane)) |
| `pdf` | PDF ingest |
| `anthropic` / `google` | Anthropic / Gemini backends |
| `turbopuffer` | Cloud vector store for the multi-tenant service |
| `agent` | pydantic-ai agent helpers |
| `all` | Everything above. Recommended. |
| `observability` | Logfire instrumentation (not in `all`) |
| `dev` | pytest, ruff, build tools (not in `all`) |
| `docs` | mkdocs-material, for building this site (not in `all`) |

```bash
pip install "membukkit[service,pdf]"
```

## Working on MemBukkit

```bash
git clone https://github.com/memseekai/membukkit.git
cd membukkit
uv sync --extra all --group dev

uv run pytest -q          # offline, no API keys needed
uv run ruff check src tests
uv run membukkit ui --demo personal-assistant
```

Only if you change the React GUI:

```bash
cd ui && npm install && npm run build   # writes src/membukkit/ui_dist
```

See [CONTRIBUTING.md](https://github.com/memseekai/membukkit/blob/main/CONTRIBUTING.md)
for the PR checklist.

## Model weights {#model-weights}

On first use MemBukkit fetches its fine-tuned bi-encoder and cross-encoder from the Hugging
Face Hub (`MemseekAI/membukkit-biencoder-v1`, `MemseekAI/membukkit-reranker-v2`) and caches
them under `~/.membukkit/models`. If the Hub is unreachable it falls back to
`all-mpnet-base-v2` and `ms-marco-MiniLM-L-6-v2`, so demos still run offline.

Resolution order: `ModelConfig(model_dir=…)` or `MEMBUKKIT_MODEL_DIR` → an explicit path →
a repo-local `models/` → the Hub → the off-the-shelf fallback.

## Troubleshooting

**`membukkit: command not found`.** The environment you installed into is not on your
`PATH`. Use `uv run membukkit …`, activate the venv, or run `python -m membukkit.cli …`.

**`the GUI needs the service extra`.** Reinstall with `[service]` or `[all]`.

**Slow first run.** The retrieval weights are downloading into `~/.membukkit/models`. Later
runs are local. Point `MEMBUKKIT_MODEL_DIR` at an existing copy to skip it.

**Docker container healthy but the GUI is unreachable.** Publish the port with
`-p 127.0.0.1:8377:8377`. The image binds `0.0.0.0` inside the container, so without `-p`
nothing reaches it. See [Docker](docker.md).

Next: **[Quickstart](quickstart.md)** · **[Demos](demos.md)** · **[Library API](library.md)**
