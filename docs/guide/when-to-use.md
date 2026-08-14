# When to use MemBukkit

Short guide by **category**, not a product bake-off. Pick the layer that matches the job.

MemBukkit’s wedge: **dated atomic facts → as-of answers → supersession → receipts**.

---

## File / markdown workspace memory

**Good for:** human-editable notes the agent (and you) can open in a folder; bootstrap preferences; daily scratch.

**Reach for MemBukkit when:** the same topic changes over time and you need **which version was true on a date**, with supersession and clickable evidence, not only “whatever is still in the file.”

Keep editable notes for bootstrap if you like; use MemBukkit when truth must be queryable as-of.

---

## Vector RAG

**Good for:** similarity search over chunks (“find passages like this question”).

**Reach for MemBukkit when:** contradictory facts accumulate (old rent, old clause, old job) and you need **supersession + as-of**, not another similar chunk that might be stale. Receipts show what the reader saw and whether evidence is current or superseded.

Plain vector retrieval stays useful for open-ended corpus search; MemBukkit is for **trusted long-term state**.

---

## Temporal graphs

**Good for:** rich relational models where entities, edges, and validity windows are first-class.

**Reach for MemBukkit when:** you want **dated facts and ask receipts** without adopting a graph stack: atomic facts, effective dates, and an ask API that returns evidence status.

---

## When not us

- Sticky preferences with **no temporal conflict** (“prefer TypeScript”) and no need for as-of or supersession badges.
- One-shot Q&A over a static PDF with no updates, where a simple chunk RAG path may be enough ([RAG mode](../RAG.md) for research-style corpus eval).
- You only need a human-editable diary and never ask “what was true in May?”

---

## Try the wedge

```bash
membukkit ui --demo personal-assistant   # same store, different as-of
membukkit ui --demo contract-qa          # amendment vs original clause
```

Next: [Agents](agents.md) · [Documents](documents.md) · [Quickstart](quickstart.md)
