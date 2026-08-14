"""Human-facing CLI commands: ingest, add, ask, chat, search, buckets, stores."""

from __future__ import annotations

import json
import sys

from membukkit.cli.common import (
    DEFAULT_LLM,
    empty_store_hint,
    format_answer,
    format_write_receipt,
    make_tqdm_progress,
    open_store,
    record_store_usage,
    resolve_as_of,
)
from membukkit.storage.localstore import LocalStore, list_stores, stores_root
from membukkit.usage import DISTILL_WARN_TOKENS, TokenUsage, estimate_cost_usd, format_cost


def cmd_add(args) -> None:
    from membukkit.memory_api import Memory

    mem, store = open_store(
        args.store,
        llm=args.llm,
        create=True,
        prompt_pack=getattr(args, "prompt_pack", None),
    )
    report = Memory.wrap(mem).add(
        args.text,
        subject=args.subject or "",
        date=args.date or None,
    )
    store.save_backend(mem.backend)
    if report.n_stored:
        store.update_meta(bucket_labels={})
        from membukkit.suggestions import refresh_store_suggestions

        refresh_store_suggestions(store, mem, llm_spec=args.llm)
    record_store_usage(store, report, args.llm)
    print(
        f"{format_write_receipt(report, llm=args.llm)}  "
        f"facts_now={mem.backend.count()}"
    )
    for w in report.warnings:
        print(f" warning: {w}", file=sys.stderr)
    if report.status == "empty_extract":
        raise SystemExit(2)


def cmd_ingest(args) -> None:
    from membukkit.ingest import parse_path

    docs = []
    for path in args.paths:
        try:
            docs.extend(parse_path(path))
        except FileNotFoundError:
            raise SystemExit(
                f"file not found: {path}\n"
                f"  tip: export WhatsApp without media, or drop a ChatGPT/Claude ZIP — "
                f"see docs/guide/bring-your-own.md\n"
                f"  example: membukkit ingest ~/Downloads/WhatsAppChat.txt --store me"
            ) from None
        except ValueError as e:
            raise SystemExit(str(e)) from None
    if not docs:
        raise SystemExit(
            "no supported files found (json/jsonl/csv/txt/md/pdf/zip with conversations.json)"
        )

    total_turns = sum(d.n_turns for d in docs)
    total_chars = sum(
        len(t["content"]) for d in docs for s in d.sessions for t in s
    )
    est_tokens = total_chars // 4
    print(f"{len(docs)} document(s), {total_turns} turns, ~{est_tokens:,} tokens")
    if est_tokens >= DISTILL_WARN_TOKENS and not args.no_distill:
        rough = estimate_cost_usd(
            TokenUsage(prompt_tokens=est_tokens, source="estimate"), args.llm
        )
        print(
            f"warning: large ingest (~{est_tokens:,} distill tokens"
            + (f", ~{format_cost(rough)}" if rough is not None else "")
            + f" with {args.llm}) — this may take a while / cost money",
            file=sys.stderr,
        )
    if args.no_distill:
        print("verbatim-only ingestion (no LLM distillation)")
    else:
        print(f"distilling atomic facts with {args.llm} (~{est_tokens:,} input tokens)")

    pack = getattr(args, "prompt_pack", None)
    mem, store = open_store(
        args.store,
        llm=args.llm,
        encoder_spec=args.encoder,
        distill=not args.no_distill,
        create=True,
        prompt_pack=pack,
    )
    if pack:
        store.update_meta(prompts=mem.prompts.to_dict(), prompt_pack=pack)
    on_progress, close_bar = make_tqdm_progress("ingest")
    n_total = 0
    n_superseded = 0
    usage_acc = TokenUsage()
    try:
        for doc in docs:
            doc_id = store.add_document(
                doc.name, doc.sessions, doc.dates, doc_type=doc.doc_type, origin=doc.origin
            )
            report = mem.ingest(
                doc.sessions,
                dates=doc.dates,
                doc_id=doc_id,
                doc_name=doc.name,
                doc_type=doc.doc_type,
                on_progress=on_progress,
            )
            n_total += int(report)
            n_superseded += len(report.superseded)
            usage_acc.merge(TokenUsage.from_dict(report.usage))
            record_store_usage(store, report, args.llm)
            line = (
                f"  + {doc.name}: {len(doc.sessions)} session(s) -> "
                f"{report.n_stored} new facts"
            )
            if report.superseded:
                line += f" · superseded {len(report.superseded)}"
            if report.status != "ok":
                line += f" · status={report.status}"
            if report.est_cost_usd is not None:
                line += f" · {format_cost(report.est_cost_usd)}"
            print(line)
            for w in report.warnings:
                print(f"    warning: {w}")
    finally:
        close_bar()
    store.save_backend(mem.backend)
    if n_total > 0:
        from membukkit.suggestions import refresh_store_suggestions

        chips = refresh_store_suggestions(store, mem, llm_spec=args.llm)
        if chips:
            print("ask chips: " + " | ".join(chips[:5]))
    print(f"store {args.store!r}: {mem.backend.count()} facts total (+{n_total} new)")
    if usage_acc.total_tokens or n_superseded:
        cost = estimate_cost_usd(usage_acc, args.llm)
        print(
            f"ingest receipt: superseded={n_superseded} · "
            f"{usage_acc.prompt_tokens:,} in / {usage_acc.completion_tokens:,} out "
            f"({usage_acc.source})"
            + (f" · {format_cost(cost)}" if cost is not None else "")
            + " — one-time index cost"
        )
    print(f"ask it something:  membukkit ask --store {args.store} \"...\"")


def cmd_ask(args) -> None:
    mem, store = open_store(
        args.store, llm=args.llm, prompt_pack=getattr(args, "prompt_pack", None)
    )
    if mem.backend.count() == 0:
        raise SystemExit(empty_store_hint(args.store))
    as_of = resolve_as_of(mem, getattr(args, "as_of", None))
    result = mem.answer(args.question, question_date=as_of)
    record_store_usage(store, result.trace, args.llm)
    print(format_answer(result, show_trace=args.show_trace, as_of=as_of))


def cmd_chat(args) -> None:
    mem, store = open_store(
        args.store, llm=args.llm, prompt_pack=getattr(args, "prompt_pack", None)
    )
    if mem.backend.count() == 0:
        raise SystemExit(empty_store_hint(args.store))
    as_of = resolve_as_of(mem, getattr(args, "as_of", None))
    print(
        f"chatting with store {args.store!r} ({mem.backend.count()} facts), "
        f"as-of {as_of}. Ctrl-D or 'exit' to quit.\n"
        f"(override with --as-of YYYY-MM-DD)"
    )
    session_cost = 0.0
    session_cost_known = False
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        result = mem.answer(question, question_date=as_of)
        record_store_usage(store, result.trace, args.llm)
        c = getattr(result.trace, "est_cost_usd", None)
        if c is not None:
            session_cost += c
            session_cost_known = True
        print(f"\nmembukkit> {format_answer(result, show_trace=args.show_trace, as_of=as_of)}")
    if session_cost_known:
        print(f"session total (est.): {format_cost(session_cost)}")


def cmd_forget(args) -> None:
    """Erase facts. Irreversible, so it previews and confirms by default."""
    mem, store = open_store(args.store, llm=args.llm)
    backend = mem.backend

    if args.doc or args.session:
        ids = backend.ids_for_source(doc_id=args.doc or "", source_session=args.session or "")
        scope = f"document {args.doc}" if args.doc else f"session {args.session}"
    else:
        try:
            ids, unknown = backend.resolve_ids(args.fact_ids)
        except ValueError as e:
            print(f"error: {e}")
            sys.exit(2)
        if unknown:
            print(f"unknown fact id(s): {', '.join(unknown)}")
        scope = f"{len(ids)} fact(s)"
    if not ids:
        print("(nothing to forget — `membukkit search` prints fact ids in column 1)")
        return

    rows = backend.list_rows_for_ids(ids)

    print(f"About to erase {scope} from store '{args.store}':\n")
    for row in rows[:10]:
        print(f"  [{row['kind']}] {row['text'][:100]}")
    if len(rows) > 10:
        print(f"  … and {len(rows) - 10} more")
    print("\nThis is irreversible, and as-of answers that relied on these will change.")
    if args.purge_source:
        print("Also erasing everything else distilled from the same turns (--purge-source).")

    if not args.yes:
        try:
            if input("Type 'yes' to continue: ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return
        except EOFError:
            print("aborted (no tty; pass --yes to skip the prompt)")
            return

    report = mem.delete_facts(ids, purge_source=args.purge_source)
    store.save_backend(backend)
    store.update_meta(bucket_labels={}, bucket_labels_lane=None)
    print(f"\nerased {report['deleted']} row(s); {backend.count()} memories remain")
    if report["revived"]:
        print(
            f"restored {len(report['revived'])} fact(s) that the deleted "
            "memories had superseded"
        )


def cmd_search(args) -> None:
    mem, store = open_store(args.store, llm=args.llm)
    as_of = (getattr(args, "as_of", None) or "").strip() or None
    result = mem.search(args.query, top_k=args.top_k, question_date=as_of)
    if not result.hits:
        print("(no matching memories)")
        return
    for hit in result.hits:
        src = f"  [{hit.doc_name} {hit.source_ref}]" if hit.doc_name else ""
        status = f"[{hit.status}] " if getattr(hit, "status", None) else ""
        print(f"{hit.ref}  {status}{hit.fact[:160]}{src}")
    t = result.trace
    if args.show_trace:
        est = getattr(t, "est_reader_tokens", 0) or 0
        print(
            f"\n--- trace: scanned {t.scan_fraction:.0%} · "
            f"{t.n_scanned}/{t.n_facts} facts"
            + (f" · ~{est} tokens" if est else "")
        )
    if args.show_source and result.hits:
        top = result.hits[0]
        if top.doc_id:
            src = store.resolve_source(top.doc_id, top.source_ref)
            if src:
                print("\n--- top hit source passage:")
                for i, turn in enumerate(src.get("turns") or []):
                    marker = ">>" if i == src.get("highlight") else "  "
                    print(f"{marker} {turn.get('role')}: {turn.get('content', '')[:200]}")


def cmd_buckets(args) -> None:
    from collections import Counter

    mem, store = open_store(args.store, llm=args.llm)
    if mem.backend.count() == 0:
        raise SystemExit(f"store {args.store!r} is empty")
    # Same lane logic as the GUI: the map is over the distilled facts;
    # verbatim is only the fallback for stores without distillation.
    lane = "atomic" if mem.backend.count_kind("atomic") > 0 else "verbatim"
    view = mem.backend.lane_view(lane) or {}
    k_eff = int(view.get("k_eff", 0))
    if not k_eff:
        print("(not enough facts to partition)")
        return
    if lane == "verbatim":
        print(
            "note: no distilled facts in this store — showing raw (verbatim) "
            "buckets. Run `membukkit distill --store "
            f"{args.store}` to extract atomic facts.",
            file=sys.stderr,
        )
    labels = {}
    if args.label:
        meta = store.meta()
        cached = meta.get("bucket_labels") or {}
        if cached and not args.relabel and meta.get("bucket_labels_lane") == lane:
            labels = {int(k): v for k, v in cached.items()}
        else:
            print(f"labeling {k_eff} buckets with {args.llm}...", file=sys.stderr)
            on_progress, close_bar = make_tqdm_progress("label")
            try:
                labels = mem.label_buckets(kind=lane, on_progress=on_progress)
            finally:
                close_bar()
            store.update_meta(
                bucket_labels={str(k): v for k, v in labels.items()},
                bucket_labels_lane=lane,
            )
    sizes = Counter(view.get("labels", []))
    print(f"{k_eff} topic buckets over {mem.backend.count_kind(lane)} {lane} facts:")
    for b in range(k_eff):
        label = labels.get(b, "")
        exemplar = (mem.backend.topic_exemplars(b, n=1, kind=lane) or [""])[0]
        print(f"  [{b:>2}] {sizes.get(b, 0):>5} facts  {label:<30} e.g. {exemplar[:80]}")


def cmd_distill(args) -> None:
    from membukkit.cli.common import distill_store

    mem, store = open_store(args.store, llm=args.llm)
    docs = store.documents()
    if not docs:
        raise SystemExit(
            f"store {args.store!r} has no preserved source documents to extract from"
        )
    before = mem.backend.count_kind("atomic")
    print(
        f"re-extracting atomic facts from {len(docs)} document(s) with {args.llm}...",
        file=sys.stderr,
    )
    on_progress, close_bar = make_tqdm_progress("distill")
    try:
        n_new = distill_store(mem, store, on_progress=on_progress)
    finally:
        close_bar()
    print(
        f"store {args.store!r}: +{n_new} new facts "
        f"({mem.backend.count_kind('atomic')} atomic now, was {before})"
    )
    if n_new:
        store.update_meta(bucket_labels={}, bucket_labels_lane=None)
        print(f"relabel the map:  membukkit buckets --store {args.store} --label")


def cmd_stores(args) -> None:
    if args.delete:
        try:
            LocalStore(args.delete, create=False).delete()
        except FileNotFoundError:
            raise SystemExit(
                f"store {args.delete!r} not found under {stores_root()}\n"
                f"  list stores:  membukkit stores"
            ) from None
        print(f"deleted store {args.delete!r}")
        return
    stores = list_stores()
    if not stores:
        print(
            f"no stores yet under {stores_root()} — create one with:\n"
            f"  membukkit add \"…\" --store notes\n"
            f"  membukkit ingest ./notes --store notes\n"
            f"  membukkit ui --demo personal-assistant"
        )
        return
    for s in stores:
        line = (
            f"{s['name']:<24} {s.get('n_facts', 0):>7} facts  "
            f"encoder={s.get('encoder', '?')}  updated={s.get('updated_at', '?')}"
        )
        totals = s.get("usage_totals") or {}
        if isinstance(totals, dict) and totals.get("est_cost_usd") is not None:
            line += f"  ~{format_cost(float(totals['est_cost_usd']))} lifetime"
        print(line)


def register(sub) -> None:
    """Attach the human-facing commands to the argparse subparsers."""
    p = sub.add_parser(
        "ingest",
        help="Ingest files (json/csv/txt/md/pdf/zip exports) into a local store",
    )
    p.add_argument("paths", nargs="+", help="files, directories, or ChatGPT/Claude export ZIPs")
    p.add_argument("--store", default="default", help="store name (default: 'default')")
    p.add_argument("--llm", default=DEFAULT_LLM, help="distiller LLM spec")
    p.add_argument("--encoder", default=None, help="encoder spec (pinned per store)")
    p.add_argument("--no-distill", action="store_true",
                   help="skip LLM distillation; store raw passages only (free, faster)")
    p.add_argument(
        "--prompt-pack",
        default=None,
        help="use-case prompt pack id or YAML path (e.g. customer_support)",
    )
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("add", help="Add a single utterance / fact to a store")
    p.add_argument("text", help="memory text to store")
    p.add_argument("--store", default="default")
    p.add_argument("--date", default=None, help="statement date YYYY-MM-DD")
    p.add_argument("--subject", default="", help="attribute the facts to this person")
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument("--prompt-pack", default=None)
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("ask", help="Ask a question against a store")
    p.add_argument("question")
    p.add_argument("--store", default="default")
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument(
        "--as-of",
        default=None,
        help="answer as of this date (YYYY-MM-DD); "
        "default: latest fact date in the store, else today",
    )
    p.add_argument("--show-trace", action="store_true",
                   help="show which buckets/memories produced the answer")
    p.add_argument("--prompt-pack", default=None, help="override store prompts for this ask")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Interactive Q&A against a store")
    p.add_argument("--store", default="default")
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument(
        "--as-of",
        default=None,
        help="answer as of this date (YYYY-MM-DD); "
        "default: latest fact date in the store, else today",
    )
    p.add_argument("--show-trace", action="store_true")
    p.add_argument("--prompt-pack", default=None)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("search", help="Retrieve memories (no answer generation)")
    p.add_argument("query")
    p.add_argument("--store", default="default")
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--as-of", default=None, help="filter as of YYYY-MM-DD")
    p.add_argument("--show-trace", action="store_true", help="print scan / token summary")
    p.add_argument("--show-source", action="store_true",
                   help="print the top hit's original source passage")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser(
        "forget",
        help="Erase memories (irreversible)",
        description=(
            "Erase facts that are wrong or that you do not want kept. Memory is "
            "append-and-supersede by default: updates keep the old fact so as-of "
            "answers work. This removes them for good, along with the verbatim "
            "turn behind each fact. Fact ids are printed in column 1 of "
            "`membukkit search`."
        ),
    )
    p.add_argument("fact_ids", nargs="*", help="fact ids to erase")
    p.add_argument("--store", default="default")
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument("--doc", help="erase everything ingested from this document id")
    p.add_argument("--session", help="erase everything from this source session")
    p.add_argument(
        "--purge-source",
        action="store_true",
        help="also erase every other fact distilled from the same turns",
    )
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("buckets", help="Inspect topic buckets (memory map)")
    p.add_argument("--store", default="default")
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument("--label", action="store_true", help="auto-label buckets with the LLM")
    p.add_argument("--relabel", action="store_true", help="force re-labeling")
    p.set_defaults(func=cmd_buckets)

    p = sub.add_parser(
        "distill",
        help="Extract atomic facts from a store's preserved raw documents "
        "(rescues verbatim-only stores; idempotent)",
    )
    p.add_argument("--store", default="default")
    p.add_argument("--llm", default=DEFAULT_LLM, help="distiller LLM spec")
    p.set_defaults(func=cmd_distill)

    p = sub.add_parser("stores", help="List (or delete) local stores")
    p.add_argument("--delete", metavar="NAME", help="delete a store")
    p.set_defaults(func=cmd_stores)
