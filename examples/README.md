# Examples

Short, copy-pasteable scripts that show how to use MemBukkit well — starting
with the “why not a vector DB?” story, then custom prompts and use-case packs.

Guides: [Agents](../docs/guide/agents.md) · [Documents](../docs/guide/documents.md) · [Customization](../docs/guide/customization.md).

MemBukkit installed? If not: [Install guide](../docs/guide/install.md).

```bash
export OPENAI_API_KEY=sk-...

python examples/00_why_membukkit.py   # HN paste target
python examples/01_personal_memory.py
python examples/04_custom_instructions.py
```

| Script | What it shows |
|--------|----------------|
| `00_why_membukkit.py` | Rent story → answer + scan% + buckets + evidence (screenshot-ready) |
| `01_personal_memory.py` | `subject=`, chat ingest, temporal “current state” ask |
| `02_support_tickets.py` | `customer_support` pack + ticket-style facts |
| `03_contracts.py` | `contracts` pack; amendment supersedes original |
| `04_custom_instructions.py` | Instruction overlays only (no full template copy) |
| `05_agent_loop.py` | tool call → `agent_ops` pack → later ask what worked |

Bundled GUI/CLI twin of the agent loop: `membukkit ui --demo agent-ops`.  
Contract demo in the GUI: `membukkit ui --demo contract-qa`.

Prompt packs live in the package (`membukkit.prompts.packs`) and load by id:

```python
from membukkit import MemorySystem
from membukkit.prompts import load_prompt_pack

mem = MemorySystem.from_pretrained(
    prompts=load_prompt_pack("customer_support"),
)
```

CLI equivalent: `membukkit ingest ./tickets.csv --store support --prompt-pack customer_support`.

Full guide: [docs/guide/customization.md](../docs/guide/customization.md).
