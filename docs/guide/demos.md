# Demos

Bundled scenarios you can load with one command. Each demo ships sample data plus
canned questions so you can see dated recall, provenance, and bucket routing
without preparing your own corpus first.

## Start here

```bash
membukkit ui --demo personal-assistant   # May vs June rent — same store, different truth
membukkit ui --demo contract-qa          # amendment beats the original clause
```

Personal assistant runs an as-of compare on Ask; contract QA is the document lane
use **Memory → Truth** for current vs superseded facts. More context:
[Quickstart](quickstart.md) · [Documents](documents.md) · [Agents](agents.md).

## Prerequisites

MemBukkit installed ([pip, uv, or Docker](install.md)) and an API key
(`export OPENAI_API_KEY=sk-...`, or paste it in the GUI). No API key? Pass a local
model instead, e.g. `--llm ollama:llama3.1`.

**First run:** downloads the [model weights](install.md#model-weights) from Hugging Face
and distills the demo with an LLM. Later `ui --demo …` reuses `demo-<name>` and
skips re-ingest.

Demo datasets ship inside the Python package (`membukkit/demos/`), so a
wheel install and an editable checkout both see the same demos.

## Commands

List the available demos:

```bash
membukkit demo --list
```

Ingest a demo and run its canned questions in the terminal:

```bash
membukkit demo personal-assistant
```

Ingest if needed, then open the GUI with that store already selected (Ask runs the
starter question so receipts appear immediately):

```bash
membukkit ui --demo personal-assistant
```

Run the terminal Q&A, then open the GUI for the same store:

```bash
membukkit demo personal-assistant --ui
```

Follow-ups on an already-loaded demo store:

```bash
membukkit chat --store demo-personal-assistant
membukkit buckets --store demo-personal-assistant --label
```

## The five demos

| Name | Title | Proves | Data format |
|------|-------|--------|-------------|
| `personal-assistant` | Personal assistant memory | Same memory, different as-of date → different truth | JSON chat history |
| `support-brain` | Customer support brain | Which ticket said that? Click through to the exact source | CSV ticket archive |
| `contract-qa` | Contract QA | Later amendment beats the original clause, with citations | Markdown contracts |
| `engineering-kb` | Engineering knowledge base | Incident ↔ ADR: see which topic buckets opened | Markdown postmortems + ADRs |
| `agent-ops` | Agent ops memory | Tool failure → fix written to memory → later ask recalls what worked | JSON ops log |

Each manifest may include `proves`, `question_date` (GUI as-of default),
`prompt_pack`, `ask_callouts`, and optional `prove_beats` (sequential as-of
compares). Personal assistant ships May vs June rent beats; contract QA keeps a
supersession callout on breach questions.

## Opening a demo later

Once a demo has been ingested, its store is named `demo-<name>` under
`~/.membukkit/stores/`.

- Run `membukkit ui` and pick `demo-*` in the sidebar, or
- Run `membukkit ui --demo <name>` again and MemBukkit reuses the existing store
  (no re-ingest) and deep-links `?store=demo-<name>&tab=ask`.

## Author your own

From a source checkout (or after installing the package, under the installed
`membukkit/demos/` tree), create a folder with a `demo.json` manifest:

```json
{
  "title": "My demo",
  "description": "What this scenario shows.",
  "proves": "One-line product claim",
  "question_date": "2024-06-01",
  "prompt_pack": "personal_assistant",
  "data": ["notes.txt"],
  "questions": [
    "What did we decide about X?"
  ],
  "prove_beats": [
    {
      "label": "as of early date",
      "question": "What did we decide about X?",
      "as_of": "2024-05-01"
    },
    {
      "label": "as of later date",
      "question": "What did we decide about X?",
      "as_of": "2024-06-01"
    }
  ],
  "ask_callouts": [
    {
      "match": "decide",
      "title": "Tip",
      "body": "Optional hint shown when the question matches."
    }
  ],
  "no_distill": false
}
```

- `data`: file names relative to that demo folder (txt / md / pdf / csv / json).
- `questions`: canned prompts for `membukkit demo <name>` and GUI suggestions.
- `no_distill` (optional): skip LLM distillation; store verbatim turns only.
- `prompt_pack` (optional): use-case pack applied on first ingest.

Use an existing manifest as a template, e.g.
[`personal-assistant/demo.json`](https://github.com/memseekai/membukkit/blob/main/src/membukkit/demos/personal-assistant/demo.json)
(also linked from the repo as `demos/personal-assistant/demo.json`).

Then:

```bash
membukkit demo my-demo
```
