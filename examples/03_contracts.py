"""Contracts: later amendment should win on conflicting terms.

Run:
    python examples/03_contracts.py

Expect: breach notification is 24 hours (addendum), not the MSA’s 48 hours.
"""

from membukkit import MemorySystem
from membukkit.prompts import load_prompt_pack

mem = MemorySystem.from_pretrained(
    llm="openai:gpt-4o-mini",
    prompts=load_prompt_pack("contracts"),
)

mem.ingest(
    sessions=[
        [{
            "role": "user",
            "content": (
                "MSA §8.2: CloudVault shall notify Meridian of a Personal Data Breach "
                "within forty-eight (48) hours of discovery."
            ),
        }],
        [{
            "role": "user",
            "content": (
                "DPA Addendum (effective 2024-09-01) amends MSA §8.2: breach notification "
                "is shortened to twenty-four (24) hours."
            ),
        }],
    ],
    dates=["2024-01-15", "2024-09-01"],
    doc_type="document",
)

res = mem.answer("How quickly must CloudVault notify Meridian of a data breach?")
print("## answer")
print(res.answer)
print()
print("## scan")
print(f"scan={res.trace.scan_fraction:.0%}")
print()
print("## evidence")
for f in (res.trace.ranked_facts or [])[:5]:
    print(" -", f)
