# Deleting memories

MemBukkit's default model is **append-and-supersede**. A knowledge update marks
the old fact superseded and keeps it, which is what makes "what was true in
May?" answerable and what puts a `superseded` badge on the receipt.

Sometimes that is not what you want. A distilled fact can simply be **wrong**,
or it can be something you would rather **not keep at all**. Those cases need
real erasure, not another layer of bookkeeping, so deletion is a first-class
operation.

It is irreversible, and it rewrites history on purpose: an as-of question that
used to return a deleted fact will no longer return it.

## Erase specific facts

```python
from membukkit import Memory

mem = Memory.from_pretrained()
r = mem.ask("How much is my rent?")
mem.delete(r.evidence[0].ref)          # the ref shown on the receipt
```

From the CLI, `membukkit search` prints the fact id in column 1:

```bash
membukkit search "rent" --store notes
# mem:bc0e3f103f50  [current] [2024-01-08] The rent is 800 EUR.

membukkit forget mem:bc0e3f103f50 --store notes
```

`forget` previews what it is about to erase and asks for confirmation. Pass
`--yes` to skip the prompt in scripts.

## What deletion actually removes

Two things happen beyond dropping the row you named.

**The verbatim source goes with it.** Every session is stored twice, as
distilled atomic facts and as the original turns. Removing only the fact would
leave the same content in the verbatim lane, still embedded and still
retrievable, which would make deletion a lie. So the source turn is removed
too, *unless another surviving fact still points at it* (one turn often
distills into several facts, and the others still need their provenance).

**Supersession is repaired.** If the fact you deleted had superseded an older
one, the older fact becomes current again. This is the case that matters most:
when the thing you are deleting is a bad correction, you want the value it
replaced to come back, not a store with no current value at all.

```python
report = mem.delete(fact_id)
report["deleted"]   # rows actually removed (fact + orphaned source)
report["revived"]   # facts that are current again
report["unknown"]   # refs that matched nothing
```

## Erase a whole source

To remove the content itself rather than one fact drawn from it:

```python
mem.delete(fact_id, purge_source=True)   # the turn + everything distilled from it
mem.forget(doc_id="…")                   # an uploaded document, both lanes
mem.forget(source_session="…")           # one ingested conversation
```

```bash
membukkit forget --doc <doc_id> --store notes
membukkit forget --session <session_id> --store notes
```

The GUI does the same thing: **Memory → Truth** lists stored facts with a
delete control per row, and removing an uploaded document from the Ingest tab
erases every fact that came from it.

## Notes and limits

- **Deletion is not exposed over MCP.** An agent that can silently erase a
  user's memory is a different risk class from one that can only append, so the
  MCP server offers `memory_add` / `memory_search` / `memory_ask` only. Erasure
  stays a human action.
- **Rebuilds are lazy.** Topic partitions, bucket labels, and the optional
  [lexical index](library.md#lexical-lane) are invalidated on delete and
  rebuilt on the next query. Nothing needs re-ingesting.
- **Deleting is not undoing an ingest.** If you re-ingest the same source, the
  facts come back. Use `forget()` on the document or session if you want the
  source gone for good.
- Stores are plain files under `~/.membukkit/stores/<name>/`, so a full wipe is
  `membukkit stores --delete <name>` or removing the directory.
