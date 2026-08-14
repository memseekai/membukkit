"""Mem0-style: overlay instructions without copying full prompt templates.

Run:
    python examples/04_custom_instructions.py

Expect: answer remembers Apex Desk / AD-200 / $349; prefers are ignored.
"""

from membukkit import MemorySystem, PromptConfig

# Only remember product SKUs and prices — ignore everything else.
prompts = PromptConfig(
    extraction_instructions=(
        "Extract ONLY product SKUs, product names, and prices the user states.\n"
        "Skip preferences, schedules, and names of people.\n"
        "If none, output NONE."
    ),
    reader_instructions="Answer with SKU and price when available; otherwise say unknown.",
)

mem = MemorySystem.from_pretrained(llm="openai:gpt-4o-mini", prompts=prompts)
mem.ingest(
    sessions=[[
        {"role": "user", "content": "I bought the Apex Desk (SKU AD-200) for $349 yesterday. Also I prefer morning meetings."},
    ]],
    dates=["2024-04-01"],
)

res = mem.answer("What did I buy and for how much?")
print("## answer")
print(res.answer)
print()
print("## evidence")
for f in (res.trace.ranked_facts or [])[:8]:
    print(" -", f)
