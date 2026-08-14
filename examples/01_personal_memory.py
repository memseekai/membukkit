"""Personal memory: named subject + temporal “current state” ask.

Run:
    python examples/01_personal_memory.py

Expect: answer cites the March raise ($2300), with scan% and top facts.
"""

from membukkit import MemorySystem
from membukkit.prompts import load_prompt_pack

mem = MemorySystem.from_pretrained(
    llm="openai:gpt-4o-mini",
    prompts=load_prompt_pack("personal_assistant"),
)

mem.ingest(
    sessions=[
        [{"role": "user", "content": "I just signed a lease — rent is $2100 at 14 Oak St."}],
        [{"role": "user", "content": "Landlord raised rent to $2300 starting March."}],
    ],
    dates=["2024-01-10", "2024-03-01"],
    subject="Alex",
)

res = mem.answer("How much is my rent now?", question_date="2024-06-01")
print("## answer")
print(res.answer)
print()
print("## scan")
print(f"scan={res.trace.scan_fraction:.0%} reader={res.trace.reader_type}")
print()
print("## evidence")
for f in (res.trace.ranked_facts or [])[:5]:
    print(" -", f)
