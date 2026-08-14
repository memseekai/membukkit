# Quickstart

Install first: [pip, uv, or Docker](install.md). Then paste an API key in the GUI (**keys**), or `export OPENAI_API_KEY=sk-...`.
No cloud key? Use a local model: add `--llm ollama:llama3.1` to any command below.

---

## Bring your own (recommended)

```bash
membukkit ui    # paste key if prompted → create a store → drop WhatsApp / ChatGPT ZIP / PDFs / CSV
```

Export steps and privacy notes: [Bring your own](bring-your-own.md).

## GUI demos (no files handy)

Two demos that show as-of truth and supersession:

```bash
membukkit ui --demo personal-assistant   # May vs June rent from the same store
membukkit ui --demo contract-qa          # amendment beats the original clause
```

**Personal assistant** opens Ask with a **May vs June** as-of compare on rent, same memory, different truth. Receipts show estimated reader tokens, superseded badges, and clickable sources.

**Contract QA** shows a later amendment winning over the original clause. Switch to **Memory → Truth** for current vs superseded facts on a timeline.

![Ask with receipts](../assets/demo-ask.gif)

![Ask with as-of truth and receipts](../assets/gui-prove-it.png)

**First run:** needs an API key (or `--llm ollama:…`), Hugging Face weight
download, and a one-time demo distill. Later opens reuse the store. See
[Install → model weights](install.md#model-weights).

Other demos: `support-brain`, `engineering-kb`, `agent-ops`. Details: [Demos](demos.md).

---

## CLI

```bash
membukkit add "I signed a lease — rent is 800€." --store notes --date 2024-01-08
# write receipt: n_stored / superseded / status
membukkit add "Landlord raised rent to 950€ from June." --store notes --date 2024-04-02
membukkit ask --store notes "How much is rent?" --as-of 2024-05-01
# → ~800€ + receipt (tokens · % of 128k · ~$)
membukkit ingest ~/Downloads/WhatsAppChat.txt --store me   # or .zip / .pdf / .csv / folder
membukkit chat --store notes --as-of 2024-05-01   # interactive; as-of defaults to latest fact
membukkit buckets --store notes --label       # topic buckets in the terminal
membukkit ui                                  # local GUI — New store → drop files on Ingest
membukkit mcp --store notes                   # MCP stdio server — leave running (needs [mcp])
```

Stores persist under `~/.membukkit/stores/<name>`. Default CLI store name is `default`; demos use `demo-<name>` (e.g. `demo-personal-assistant`).

---

## Python API (agent facade)

```python
from membukkit import Memory

mem = Memory.from_pretrained(llm="openai:gpt-4o-mini")
w = mem.add("I switched to a vegan diet last month.", subject="me", date="2024-06-01")
print(w.status, w.n_stored)  # write receipt
mem.add("My manager is Dana; we have morning standups.", subject="me", date="2024-06-10")
r = mem.ask("What diet am I on?", as_of="2024-07-01")
print(r.answer, r.est_reader_tokens)
```

Power users: `MemorySystem.ingest` / `.answer` (sessions + full config).

```python
from membukkit import MemorySystem

mem = MemorySystem.from_pretrained(llm="openai:gpt-4o-mini")

mem.ingest(
    sessions=[
        [{"role": "user", "content": "I switched to a vegan diet last month."}],
        [{"role": "user", "content": "My manager is Dana; we have morning standups."}],
    ],
    dates=["2024/06/01", "2024/06/10"],
)

res = mem.answer("What diet is the user on now?", question_date="2024/07/01")
print(res.answer)                 # dated, self-contained answer
print(res.trace.scan_fraction)    # fraction of memory scanned
print(res.trace.opened_buckets)   # which topic buckets fired
print(res.facts)                  # evidence lines the reader saw
```

Weights resolve automatically (HF Hub → cache → off-the-shelf fallback). See [Install → model weights](install.md) and [Library API](library.md).

---

## Reproduce a benchmark

```bash
membukkit bench --list
membukkit bench --repro longmemeval-gpt4o-mini --lite   # cheap smoke
```

Full recipes and claimed numbers: [Benchmarks](benchmarks.md).

---

## Next steps

- **[Agents](agents.md)**: turn loop, write receipts, curl / MCP
- **[Documents](documents.md)**: ingest files, citations, contract demo
- **[Demos](demos.md)**: bundled scenarios and authoring your own
- **[MCP](mcp.md)**: Cursor / Claude Desktop
- **[When to use](when-to-use.md)**: file memory vs vector RAG vs temporal graphs
- **[Library API](library.md)**: configuration, buckets, traces
- **[Install](install.md)**: extras, git vs clone, contributor setup
