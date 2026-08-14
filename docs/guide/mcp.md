# MCP server

MemBukkit ships a thin [Model Context Protocol](https://modelcontextprotocol.io/) server so
Cursor, Claude Desktop, and other MCP clients can call the same memory facade as the SDK.

Same local stores under `~/.membukkit/stores/` as the CLI and GUI. Agent turn-loop patterns: [Agents](agents.md).

## Install

```bash
pip install -e ".[mcp]"
# or
pip install -e ".[all]"
```

## Tools

| Tool | Purpose |
|------|---------|
| `memory_add` | Store an utterance (`text`, optional `date`, `subject`, `store_name`) |
| `memory_search` | Retrieve evidence (`query`, optional `as_of`, `top_k`, `store_name`) |
| `memory_ask` | Answer with receipts (`query`, optional `as_of`, `store_name`) |

Default store comes from `--store` or the `MEMBUKKIT_STORE` environment variable.

## Run

```bash
membukkit mcp --store notes   # blocks on stdio — leave it running for your MCP client
membukkit mcp --list-tools    # JSON catalog (exits)
```

The process waiting without further output is normal; Cursor/Claude Desktop attach to that stdio pipe.

## Cursor example

```json
{
  "mcpServers": {
    "membukkit": {
      "command": "membukkit",
      "args": ["mcp", "--store", "notes"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "MEMBUKKIT_STORE": "notes"
      }
    }
  }
}
```

Point `command` at your venv’s `membukkit` if it is not on `PATH`. Restart Cursor after editing MCP config so tools appear.

### Example prompts

Once tools are connected, try:

- “Remember that my rent is 800€ as of January 2024.” → should call `memory_add` (with a date when possible).
- “Landlord raised rent to 950€ from June 2024, store that.” → another `memory_add`; later facts can supersede earlier ones.
- “How much is my rent as of May 2024?” → `memory_ask` with `as_of` around `2024-05-01`.
- “Search memory for lease / rent facts.” → `memory_search`.

Prefer `memory_ask` when you want a natural-language answer plus receipts; use `memory_search` when the agent should pack evidence into its own prompt.

## Claude Desktop

Same idea: add a stdio server entry with `command` / `args` / `env` as above (see Claude Desktop’s MCP docs for the config file location on your OS).

## Failure notes

| Symptom | Likely cause |
|---------|----------------|
| Tools missing / server won’t start | Package not installed with `[mcp]` / `[all]`; wrong `command` path |
| Add/ask errors mentioning API key | `OPENAI_API_KEY` (or your LLM env) not set in the MCP `env` block |
| Empty or weak answers | Store has no facts yet: `memory_add` first, or ingest via CLI/GUI |
| `empty_extract` / nothing stored | Distiller produced no facts; check the write receipt fields, retry with clearer text + `date` |
| Wrong store | Set `--store` / `MEMBUKKIT_STORE` to the same name you use in the GUI |

List tools without a client: `membukkit mcp --list-tools`.

## Next

- [Agents](agents.md): recall → write → ask as-of
- [Documents](documents.md): file ingest + citations
- [Quickstart](quickstart.md)
