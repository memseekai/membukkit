# Bring your own data

Drop files you already have. No need to invent a sample corpus. Everything stays
local under `~/.membukkit/stores/`.

## Three paths

| Who | What to drop | Export |
|---|---|---|
| Anyone with a phone | One WhatsApp chat | Chat → Export chat → **Without media** → `.txt` / `_chat.txt` |
| ChatGPT / Claude users | Data export ZIP | Settings → Export data → unzip or drop the ZIP |
| Work / builders | PDFs, CRM CSV, Notion/Obsidian folder, project docs | See below |

Then:

```bash
membukkit ui                    # paste API key if prompted → New store → Ingest → drop files
# or from a real export path on your machine:
membukkit ingest ~/Downloads/WhatsAppChat.txt --store me
membukkit ingest ~/Downloads/chatgpt-export.zip --store me
membukkit ask --store me "How much is rent?" --as-of 2024-05-01
```

No shell key? Use **keys** in the GUI sidebar (saved under `~/.membukkit/credentials.env`,
mode `0600`). The CLI loads the same file when `OPENAI_API_KEY` is unset.

Ask receipts show **reader tokens**, **% of a 128k window**, **scan fraction**, and
an estimated **$** (metered from the provider when available, otherwise chars÷4).

---

## WhatsApp

1. Open **one** conversation (not “all chats”).
2. Export **without media**.
3. Drop the `.txt` (or unzip and use `_chat.txt`) into the GUI.

MemBukkit detects WhatsApp timestamps and buckets messages by day.

**Privacy:** pick a chat you are comfortable storing on your laptop.

---

## ChatGPT

1. chatgpt.com → Settings → **Data controls** → **Export data**.
2. Download the email ZIP (link expires ~24h).
3. Drop the ZIP, or `conversations.json` / `conversations-000.json`, into MemBukkit.

Large exports can be expensive to distill once. The CLI warns above ~2M estimated
tokens; prefer a smaller export or a subset of chats when trying the product.

---

## Claude

1. claude.ai (desktop) → Settings → **Privacy** → **Export data**.
2. Download the ZIP → drop it or `conversations.json`.

---

## Work files / CRM

- **PDF / Markdown / TXT**: contracts, decks, notes (`pip install` with `[pdf]` or `[all]` for PDF).
- **CSV**, HubSpot/Salesforce deals or tickets export (columns like `created_at` / `date` help dating).
- **Notion**. Settings → Export → **Markdown & CSV** → unzip → ingest the folder.
- **Obsidian / project**: point at a vault subfolder or `./docs ./README.md`.

```bash
membukkit ingest ./deals.csv ./msa.pdf ./amendment.pdf --store sales
membukkit ingest ./Export-Notion --store notes
membukkit ingest ./README.md ./docs --store project
membukkit mcp --store project   # leave running for Cursor
```

---

## Cost notes

- **Ingest / distill** is the main one-time cost (builds atomic facts).
- **Ask** usually costs far less, only the ranked slice reaches the reader.
- Receipts label **metered** (API usage) vs **est.** (chars÷4). Not your provider invoice.

---

## Ingest now, distill later

Distillation is the part of ingest that calls an LLM, and on a large export it is
the part that costs money and takes time. You can skip it:

```bash
membukkit ingest ./big-export --store me --no-distill
```

That stores every turn verbatim and does **no distillation**, which is where the
per-session LLM cost lives; the ingest receipt comes back at `$0`. The store is
searchable immediately, because the verbatim lane is indexed like any other. It
also works with no API key at hand: the CLI still tries one small call afterwards
to suggest starter questions, and simply skips the suggestions if it cannot.

What you give up is the atomic lane, and with it the things built on dated facts:
supersession, as-of answers, and the fact-level receipts. Ask still works, but it
answers from raw turns.

When you are ready, distil the raw documents the store kept:

```bash
membukkit distill --store me
```

It reads the sources the store preserved, extracts atomic facts, and links
supersessions, turning a verbatim-only store into a full one. A store that was
ingested normally has nothing to do and says so.

Running it twice is safe but not free. Fact ids hash on text, date, and kind, so
re-running adds no duplicates. It does re-call the LLM for every session, though,
because the distiller's cache lives in the process rather than on disk. Treat a
second run as a second bill, not a resume.

---

## No files handy?

```bash
membukkit ui --demo personal-assistant
membukkit ui --demo contract-qa
```

See [Demos](demos.md) · [Quickstart](quickstart.md) · [MCP](mcp.md).
