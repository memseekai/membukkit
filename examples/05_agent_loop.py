"""Minimal agent memory: record a tool failure, ask what worked next time.

Run:
    python examples/05_agent_loop.py

Expect: answer recalls the longer kubectl timeout that fixed deploy.
"""

from membukkit import MemorySystem
from membukkit.prompts import load_prompt_pack

mem = MemorySystem.from_pretrained(
    llm="openai:gpt-4o-mini",
    prompts=load_prompt_pack("agent_ops"),
)


def run_tool(name: str) -> str:
    if name == "deploy":
        return "ERROR: kubectl timeout talking to staging cluster"
    if name == "deploy_retry":
        return "ok — used --request-timeout=60s"
    return "ok"


# Turn 1: tool fails, agent notes it into memory.
err = run_tool("deploy")
mem.ingest(
    sessions=[[{"role": "user", "content": f"Tool deploy failed: {err}. Will retry with longer timeout."}]],
    dates=["2024-07-01"],
    doc_type="document",
)

# Turn 2: retry works — store the durable lesson.
ok = run_tool("deploy_retry")
mem.ingest(
    sessions=[[{"role": "user", "content": f"Tool deploy_retry succeeded: {ok}."}]],
    dates=["2024-07-01"],
    doc_type="document",
)

# Later run: ask memory before trying again.
res = mem.answer("Last time deploy failed — what fallback worked?")
print("## answer")
print(res.answer)
print()
print("## evidence")
for f in (res.trace.ranked_facts or [])[:5]:
    print(" -", f)
