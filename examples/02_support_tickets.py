"""Support brain: customer_support pack over synthetic tickets.

Run:
    python examples/02_support_tickets.py

Expect: answer points at the multipart chunk-size fix from ticket #512.
"""

from membukkit import MemorySystem
from membukkit.prompts import load_prompt_pack

mem = MemorySystem.from_pretrained(
    llm="openai:gpt-4o-mini",
    prompts=load_prompt_pack("customer_support"),
)

# Document-mode sessions (no assistant role) → document extraction prompt.
mem.ingest(
    sessions=[
        [{"role": "user", "content": "Ticket #441 Aurora Foods: export timeout to S3. Agent Maya rotated the IAM key; exports succeeded."}],
        [{"role": "user", "content": "Ticket #512 Aurora Foods: export timeout again after key rotation policy. Fix: bump multipart chunk size to 64MB."}],
    ],
    dates=["2024-05-02", "2024-06-18"],
    doc_type="document",
)

res = mem.answer("Aurora Foods has export timeouts again — what fixed it before?")
print("## answer")
print(res.answer)
print()
print("## scan")
print(f"scan={res.trace.scan_fraction:.0%}")
print()
print("## evidence")
for f in (res.trace.ranked_facts or [])[:5]:
    print(" -", f)
