# Docker

Run the GUI (and its local HTTP API) in a container, no Python environment on the host.

## Run

```bash
docker run -p 127.0.0.1:8377:8377 -v membukkit-data:/data \
  -e OPENAI_API_KEY ghcr.io/memseekai/membukkit
```

Then open **http://localhost:8377**. What the flags do:

| Flag | Why |
|---|---|
| `-p 127.0.0.1:8377:8377` | Publish the GUI + local API. Bind to `127.0.0.1`: the local app has **no auth**, so don't expose it beyond your machine. |
| `-v membukkit-data:/data` | Persist stores, model weights, and saved keys across container restarts (see below). |
| `-e OPENAI_API_KEY` | Pass your key through from the host env. Or skip it and paste the key in the GUI: it's saved into the volume. |

Image tags: `latest` (latest release), `vX.Y.Z` (pinned release), `main` (latest main build).
Building locally instead:

```bash
git clone https://github.com/memseekai/membukkit.git && cd membukkit
docker build -t membukkit .
docker run -p 127.0.0.1:8377:8377 -v membukkit-data:/data -e OPENAI_API_KEY membukkit
```

## The `/data` volume

The image sets `MEMBUKKIT_HOME=/data`, so everything MemBukkit persists lands in one volume:

| Path | Contents |
|---|---|
| `/data/stores/` | Your memory stores (facts, embeddings, metadata). |
| `/data/models/` | Encoder/reranker weights, downloaded from Hugging Face on first ask (hundreds of MB: the volume makes this a one-time cost). |
| `/data/credentials.env` | API keys pasted in the GUI's **keys** panel. |

Skip the `-v` and all of the above is discarded with the container.

## Environment

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Default LLM provider (distillation + answering). |
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Alternative providers (pass `--llm` accordingly). |
| `COMPAT_BASE_URL` + `COMPAT_API_KEY` | Any OpenAI-compatible host (Together, Groq, OpenRouter, …). |
| `OLLAMA_HOST` | Local models. From inside a container, point at the host: `-e OLLAMA_HOST=http://host.docker.internal:11434` (on Linux add `--add-host=host.docker.internal:host-gateway`). |

## Compose

The repo ships a [docker-compose.yml](https://github.com/memseekai/membukkit/blob/main/docker-compose.yml):

```yaml
services:
  membukkit:
    image: ghcr.io/memseekai/membukkit:latest
    build: .
    ports:
      - "127.0.0.1:8377:8377"
    volumes:
      - membukkit-data:/data
    environment:   # bare names pass the host's value through; unset stays unset
      - OPENAI_API_KEY
      - ANTHROPIC_API_KEY
      - GEMINI_API_KEY
      - OLLAMA_HOST
volumes:
  membukkit-data:
```

```bash
export OPENAI_API_KEY=sk-...
docker compose up
```

## Ingesting files

The GUI's drag-and-drop upload works as-is, files travel over HTTP into the container. To use
`membukkit ingest` with paths instead, bind-mount the folder read-only:

```bash
docker run --rm -v membukkit-data:/data -v ~/Downloads:/import:ro \
  -e OPENAI_API_KEY ghcr.io/memseekai/membukkit \
  membukkit ingest /import/WhatsAppChat.txt --store me
```

## Multi-tenant service instead of the GUI

The default command runs the local single-user GUI. For the Turbopuffer-backed
[multi-tenant service](service.md), override it:

```bash
docker run -p 127.0.0.1:8080:8080 -v membukkit-data:/data \
  -e OPENAI_API_KEY -e TURBOPUFFER_API_KEY -e TURBOPUFFER_REGION \
  ghcr.io/memseekai/membukkit \
  membukkit serve --host 0.0.0.0 --port 8080
```

Note the health endpoint differs: the GUI answers on `/api/health`, the service on `/health`
(the image's built-in `HEALTHCHECK` assumes the GUI, override or disable it for `serve`).
