# Customization: prompts & recipes

MemBukkit ships strong default extraction and reader prompts. For a real product
you usually want to **steer what gets remembered** and **how answers are
phrased**, without forking the library.

Two layers:

1. **Instruction overlays** (Mem0-style), short natural-language rules appended
   to the built-in templates. Lowest friction.
2. **Full template overrides**, replace an entire prompt. Use when you need a
   different output contract or structure.

Both live on [`PromptConfig`](https://github.com/memseekai/membukkit/blob/main/src/membukkit/config.py)
and are honored by `MemorySystem.from_pretrained`, the CLI (`--prompt-pack`),
and the GUI **Prompts** tab.

Frozen benchmark recipes (`membukkit bench --repro …`) always use the built-in
defaults so claimed scores stay honest.

---

## Quick start: instruction overlays

```python
from membukkit import MemorySystem, PromptConfig

mem = MemorySystem.from_pretrained(
    prompts=PromptConfig(
        extraction_instructions=(
            "Extract ONLY order numbers, SKUs, and ship dates. "
            "Skip chitchat. If none, output NONE."
        ),
        reader_instructions="Answer with the order id and date when known.",
    ),
)
```

Changing **extraction** instructions affects new ingests / `distill` runs only
existing facts are not rewritten. Re-ingest or use **Extract atomic facts** in
the GUI after a change.

---

## Use-case prompt packs

Shipped packs are YAML files under `membukkit/prompts/packs/`. Load by id:

```python
from membukkit.prompts import load_prompt_pack, list_prompt_packs

print([p["id"] for p in list_prompt_packs()])
mem = MemorySystem.from_pretrained(prompts=load_prompt_pack("customer_support"))
```

| Pack id | Tuned for |
|---------|-----------|
| `personal_assistant` | Consumer chat: preferences, schedules, money, knowledge updates |
| `customer_support` | Tickets / CRM: customer, symptom, fix; skip chitchat |
| `contracts` | Legal / policy: SLAs, caps, notice periods; amendments win |
| `engineering_kb` | Postmortems / ADRs: root cause, decisions, owners |
| `agent_ops` | Coding agents: tool failures, retries, durable prefs |

CLI:

```bash
membukkit ingest ./tickets.csv --store support --prompt-pack customer_support
membukkit ask --store support "What fixed Aurora's export timeouts?"
membukkit ui --demo support-brain   # then open the Prompts tab to tweak
```

Or point at your own file: `--prompt-pack ./my_pack.yaml`.

---

## Full template overrides

Every override must keep the placeholders the pipeline fills in:

| Field | Required placeholders |
|-------|------------------------|
| `extraction` | `{date}`, `{transcript}` |
| `extraction_named` | `{subject}`, `{date}`, `{transcript}` |
| `extraction_document` | `{date}`, `{transcript}` |
| `dated_reader` / `reasoning_reader` / `recommendation_reader` | `{identity_preamble}`, `{today_line}`, `{fact_block}`, `{question}` |

```python
prompts = PromptConfig(
    extraction=(
        "List facts as `index | fact` lines.\n"
        "Conversation (occurred on {date}):\n{transcript}\n\nFacts:"
    ),
)
```

If you set a full reader template, `reader_instructions` is **not** also
appended (avoids double-annotation). Instruction-only packs use the stock
template + overlay.

Distill caches key on the active extraction template: custom prompts never
reuse the default cache entries.

---

## Recipes (use case → 3 commands)

**Personal assistant**

```bash
membukkit ingest chat.json --store me --prompt-pack personal_assistant
membukkit ask --store me "How much is my rent now?"
membukkit ui   # Prompts tab → tweak instructions
```

**Support brain**

```bash
membukkit ingest tickets.csv --store support --prompt-pack customer_support
membukkit ask --store support "What fixed Aurora Foods last time?" --show-trace
```

**Contracts**

```bash
membukkit ingest ./msa.md ./dpa.md --store legal --prompt-pack contracts
membukkit ask --store legal "What is the breach notification window?"
```

**Agent ops**

```bash
python examples/05_agent_loop.py
```

More copy-paste scripts: [`examples/`](https://github.com/memseekai/membukkit/tree/main/examples).

---

## Other knobs that matter

| Knob | Why |
|------|-----|
| `subject="Alex"` on `ingest` | Named-person extraction (`extraction_named`) |
| `doc_type="document"` / no assistant role | Document / multi-speaker extraction prompt |
| `RetrievalConfig(scan_budget=0.5)` | Scan more of memory for harder questions |
| `RetrievalConfig(num_buckets=12)` | Fewer buckets for small stores |
| Local store under `~/.membukkit/stores/` | Persists facts **and** saved `prompts` from the GUI/CLI |

See [Library API](library.md) and [Install](install.md).

---

## Edit prompts in the GUI

1. `membukkit ui` (or `membukkit ui --demo personal-assistant`)
2. Select a store → **prompts** tab
3. Apply a pack, or edit **Instructions** (default) / **Advanced** full templates
4. **Save**, persisted on the store; next ingest/ask uses them
5. If you changed extraction rules, re-run **Extract atomic facts** (or re-ingest)

---

## Author your own pack

```yaml
title: My vertical
description: What this pack is for.
extraction_instructions: |
  Extract only …
reader_instructions: |
  Answer like …
```

Save as `my_vertical.yaml` and pass `--prompt-pack ./my_vertical.yaml`, or drop
it into a checkout under `src/membukkit/prompts/packs/` to ship it with the
package.
