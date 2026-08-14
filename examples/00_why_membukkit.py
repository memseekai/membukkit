"""Why MemBukkit: dated facts, supersession, scan budget, evidence.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/00_why_membukkit.py

Expect (shape, not exact wording):
    ## write
    stored N · superseded 1 · status=ok
    ## answer
    Rent is $2300 … (after the March raise from $2100)
    ## scan
    scanned ~N% of memory · reader=…
    ## evidence
     - [current] … rent … $2300 …
     - [superseded] … rent … $2100 …
"""

from membukkit import Memory

mem = Memory.from_pretrained(llm="openai:gpt-4o-mini")

r1 = mem.add(
    "I just signed a lease — rent is $2100 at 14 Oak St.",
    subject="Alex",
    date="2024-01-10",
)
r2 = mem.add(
    "Landlord raised rent to $2300 starting March.",
    subject="Alex",
    date="2024-03-01",
)

print("## write")
print(
    f"stored {r1.n_stored + r2.n_stored} · "
    f"superseded {len(r1.superseded) + len(r2.superseded)} · "
    f"status={r2.status}"
)
for w in r1.warnings + r2.warnings:
    print(" warning:", w)
print()

res = mem.ask(
    "How much is my rent, and when did it change?",
    as_of="2024-06-01",
)

print("## answer")
print(res.answer or "(no answer)")
print()
print("## scan")
print(
    f"scanned {res.scan_fraction:.0%} of memory · "
    f"reader={res.reader_type} · "
    f"facts={res.n_scanned}/{res.n_facts} · "
    f"~{res.est_reader_tokens} reader tokens"
)
print()
print("## evidence")
for e in res.evidence[:8]:
    print(f" - [{e.status}] {e.fact}")
