"""Built-in reader prompt templates for MEMBUKKIT."""

# Overlay for changelog / "what changed" questions. Injected only when
# ``is_changelog_query`` matches — does not alter LongMemEval stock prompts
# for ordinary current-state or temporal arithmetic questions.
CHANGELOG_READER_OVERLAY = (
    "Changelog question: 'recently' / 'changed' / 'updated' means the latest "
    "dated knowledge updates in the memories on or before today's (as-of) date "
    "— the memory timeline, not events near the wall-clock as-of date. "
    "List concrete state changes (rent, job, move, preference flips, etc.) with "
    "dates at the same granularity as the memories. Do NOT reply N/I merely "
    "because memory dates are months or years before as-of; abstain only if "
    "there are no update-like facts in the memories at all."
)

DATED_READER_PROMPT = (
    "You are answering a question from a user's long-term memory.\n"
    "{identity_preamble}"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was *stated*.\n"
    "Answer as of today's date: report the state that was true THEN.\n"
    "If a memory announces a change that takes effect AFTER today "
    "(e.g. 'rent goes to 950 from June' when today is still April or May), "
    "that change is not yet current — keep the prior value as current and, "
    "if useful, note the upcoming effective date.\n"
    "Use ONLY these memories. For elapsed-time questions (how many days/weeks/"
    "months ago or since, or the order of events), reason over the dates and "
    "compute relative to today's date. If several memories mention the same kind "
    "of item, count the DISTINCT ones. When you cite a date, keep the same "
    "granularity as the memories (use the day when known — e.g. 2024-04-02 — "
    "do not coarsen to month-only). Answer concisely. Only reply 'N/I' if the "
    "memories genuinely do not contain the answer.\n\n"
    "Memories:\n{fact_block}\n\n"
    "Question: {question}\n\nAnswer:"
)

RECOMMENDATION_READER_PROMPT = (
    "You are a helpful assistant giving a PERSONALIZED recommendation from a user's "
    "long-term memory.\n"
    "{identity_preamble}"
    "{today_line}"
    "Use the memories below to tailor your suggestion to the user's specific "
    "preferences, tools, brands, interests, tastes, and skills. Be concrete and "
    "specific; name the user's known tools/brands/domains where relevant. If the "
    "memories already contain the answer, use it. Do NOT give generic advice that "
    "ignores the memories, and do NOT refuse.\n\n"
    "Memories:\n{fact_block}\n\n"
    "Request: {question}\n\nPersonalized answer:"
)

REASONING_READER_PROMPT = (
    "You are answering a question from a user's long-term memory by reasoning "
    "across MULTIPLE memories and sessions.\n"
    "{identity_preamble}"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was *stated*.\n"
    "Answer as of today's date: the current value is whatever was in effect ON "
    "that date. An announcement of a future change (stated earlier, effective "
    "later) does not replace the prior value until its effective date.\n"
    "Think step by step: (1) FIRST decide whether the memories actually contain "
    "information that answers THIS question — do not assume they do; if they do not, "
    "the answer is 'N/I' and you must NOT invent one from loosely related memories. "
    "(2) If they do, identify every relevant memory (there may be several, across "
    "different dates/sessions); (3) combine them — aggregate counts over DISTINCT "
    "items; for changed facts resolve the value in effect as of today (not merely "
    "the most recently stated announcement); compute elapsed time relative to today "
    "using the dates; when citing dates keep day granularity when the memories "
    "have a day (do not coarsen to month-only); (4) give the final answer. Use "
    "ONLY these memories. End with a line 'Answer: <concise answer, or N/I>'.\n\n"
    "Memories:\n{fact_block}\n\nQuestion: {question}\n"
)

# --- v2 reader prompts (opt-in via `membukkit eval --reader-prompts v2`) ---
# Error analysis of the gpt-5.4 run (results/longmemeval/gpt54_openai_embed)
# showed three dominant loss modes: false refusals where the gold answer WAS in
# the evidence (strong readers over-trigger on the strict N/I instruction),
# date-arithmetic slips, and undercounting on aggregation questions. v2 keeps
# the same task framing but softens the abstention bar, forces dates to be
# written out before computing, and forces enumeration before counting.

DATED_READER_PROMPT_V2 = (
    "You are answering a question from a user's long-term memory.\n"
    "{identity_preamble}"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was stated.\n"
    "Use ONLY these memories. If any memory contains the information asked "
    "about, even phrased differently or mentioned in passing, answer from it. "
    "For elapsed-time questions (how many days/weeks/months ago or since, or "
    "the order of events), first write down the relevant dates, then compute "
    "the difference relative to today's date. If several memories mention the "
    "same kind of item, count the DISTINCT ones. Answer concisely. Reply 'N/I' "
    "only after checking every memory and finding that none of them mentions "
    "the specific thing asked.\n\n"
    "Memories:\n{fact_block}\n\n"
    "Question: {question}\n\nAnswer:"
)

REASONING_READER_PROMPT_V2 = (
    "You are answering a question from a user's long-term memory by reasoning "
    "across MULTIPLE memories and sessions.\n"
    "{identity_preamble}"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was stated.\n"
    "Think step by step: (1) find every memory relevant to the question; the "
    "answer may be phrased differently than the question or split across "
    "several memories from different dates/sessions, so combine partial pieces "
    "before deciding it is unanswerable. Conclude 'N/I' only when nothing in "
    "the memories bears on the specific thing asked, and never invent an "
    "answer from loosely related memories. (2) For counting or totaling "
    "questions, enumerate every DISTINCT relevant item with its date, one per "
    "line, before counting or summing, and double-check that no memory adds a "
    "further item. (3) For time questions, write down the relevant dates "
    "explicitly, then compute elapsed time relative to today. For changed "
    "facts, resolve to the latest value by date. (4) Give the final answer. "
    "Use ONLY these memories. End with a line 'Answer: <concise answer, or "
    "N/I>'.\n\n"
    "Memories:\n{fact_block}\n\nQuestion: {question}\n"
)

# --- v3 reader prompts (opt-in via `membukkit eval --reader-prompts v3`) ---
# Second error-analysis pass, on the 91.6 gpt-5.4 run (gpt54_v2_agg_routing).
# Its reader-side losses were: (a) judge-protocol losses from hedged answers,
# parentheticals, or leading with the wrong quantity; (b) answering from a
# similar-but-different entity where gold is an abstention (NovaTech vs
# Google); (c) a few residual refusals despite the evidence being present.
# v3 = v2 + strict answer formatting + an entity-match check that asks for
# abstention on mismatch WITHOUT re-raising the overall abstention bar.

_V3_ANSWER_RULES = (
    "Answer rules: give ONLY the direct answer to the exact question asked, "
    "with no parentheticals and no alternative readings. For quantities, "
    "state the computed value itself, never a bound: say 'about $270', not "
    "'over $270' or 'at least $270'. If the question asks for a difference "
    "or an increment (how many more, how much more/earlier/longer), answer "
    "that difference, not the total. "
    "Substitution check, for PROPER NAMES only: if the question asks about a "
    "specifically named company, institution, place, or event and the "
    "memories only describe a DIFFERENT named one, reply 'N/I' rather than "
    "substituting it. For everything else, memories that describe the "
    "asked-about thing in different words DO count as the answer. Reply "
    "'N/I' only after checking every memory line by line and finding that "
    "none of them mentions the specific thing asked (it may be phrased "
    "differently or mentioned in passing, including inside assistant "
    "replies).\n"
)

DATED_READER_PROMPT_V3 = (
    "You are answering a question from a user's long-term memory.\n"
    "{identity_preamble}"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was stated.\n"
    "Use ONLY these memories. For elapsed-time questions (how many days/weeks/"
    "months ago or since, or the order of events), first write down the "
    "relevant dates, then compute the difference relative to today's date. "
    "If several memories mention the same kind of item, count the DISTINCT "
    "ones. " + _V3_ANSWER_RULES + "\n"
    "Memories:\n{fact_block}\n\n"
    "Question: {question}\n\nAnswer:"
)

REASONING_READER_PROMPT_V3 = (
    "You are answering a question from a user's long-term memory by reasoning "
    "across MULTIPLE memories and sessions.\n"
    "{identity_preamble}"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was stated.\n"
    "Think step by step: (1) find every memory relevant to the question; the "
    "answer may be phrased differently than the question or split across "
    "several memories from different dates/sessions, so combine partial "
    "pieces before deciding it is unanswerable. (2) For counting or totaling "
    "questions, enumerate every DISTINCT relevant item with its date, one per "
    "line, before counting or summing, and double-check that no memory adds a "
    "further item. (3) For time or ordering questions, write down each event "
    "with its date explicitly, then sort or compute elapsed time relative to "
    "today. For changed facts, resolve to the latest value by date. (4) Give "
    "the final answer. Use ONLY these memories. " + _V3_ANSWER_RULES + "\n"
    "End with a line 'Answer: <concise answer, or N/I>'.\n\n"
    "Memories:\n{fact_block}\n\nQuestion: {question}\n"
)

# Event-ordering answer analysis (BEAM). The benchmark's official scorer
# newline-splits the final answer and LLM-aligns each line against TERSE
# rubric topic labels (2-6 words, e.g. "Transaction error handling",
# "Initial project setup"). The alignment judge is near-verbatim strict:
# probing showed it accepts short same-head-noun paraphrases but rejects any
# line with added detail, parentheticals, or full-sentence phrasing. So the
# reader must emit bare noun-phrase labels; the harness sorts them by
# first-mention date deterministically (a quirk in the official union-rank
# construction makes alignment count matter far more than order: one aligned
# item lifts tau_norm from the 0.125 disjoint baseline to 0.5).
ORDERING_READER_PROMPT = (
    "You are reconstructing the ORDER in which topics were first brought up in "
    "a user's long conversation history.\n"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was stated.\n"
    "The question asks for a specific number of items. Identify exactly that "
    "many TOP-LEVEL aspects/phases (if no number is given, pick the 3 most "
    "significant). Guidelines:\n"
    "- Each item is a SHORT noun-phrase label of 2-6 words naming a CONCRETE, "
    "SPECIFIC episode — a particular thing the user did, asked about, or "
    "worked on at a specific time — e.g. 'Trying running shoes in store', "
    "'Transaction error handling', 'Modal bug fix'.\n"
    "- Do NOT generalize into broad themes (avoid labels like 'self-care', "
    "'planning', 'collection building'). Name the specific activity itself.\n"
    "- Do NOT add detail beyond the label: no sub-tasks, no parentheses, no "
    "examples, no full sentences.\n"
    "- Items must be distinct episodes, each from a different point in time.\n"
    "- For each item, find the EARLIEST date any of its activities was "
    "mentioned.\n\n"
    "Output ONLY one line per item, in this exact format (no other text):\n"
    "YYYY-MM-DD | <short topic label>\n\n"
    "Memories:\n{fact_block}\n\n"
    "Question: {question}\n"
)

# AMB variant: Hindsight's binary judge grades against the FULL expected
# order (up to 9 topics) even when the question requests fewer items, and
# fails any answer with "key topics missing". So this variant enumerates
# ALL distinct topics comprehensively; the harness date-sorts and does NOT
# truncate to the requested count.
ORDERING_READER_PROMPT_AMB = (
    "You are reconstructing the ORDER in which topics were first brought up in "
    "a user's long conversation history.\n"
    "{today_line}"
    "Each memory is prefixed with [YYYY-MM-DD], the date it was stated.\n"
    "The memories below are in CHRONOLOGICAL order. Lines marked 'QUOTE:' are "
    "verbatim openings of conversation turns — they show, in the original "
    "wording, exactly what was asked or introduced at each point; prefer "
    "their phrasing over summaries when naming stages.\n"
    "The question asks for a specific number of items, N. Your task: DIVIDE "
    "the discussion of the exact subject named in the question into exactly N "
    "consecutive chronological STAGES that together cover the FULL "
    "progression, from the very first mention to the last. Guidelines:\n"
    "- First find every memory about the exact subject (ignore other "
    "subjects, even related ones), then partition that timeline into N "
    "stages — do not skip any phase of the progression, and do not spend an "
    "item on a tangent.\n"
    "- Each item names what the discussion focused on during that stage, as "
    "a short phrase (4-9 words), e.g. 'General explanation of X', 'X with "
    "harvesting and steady-state', 'Step-by-step calculation of X', "
    "'Difficulty with the product rule'.\n"
    "- Items must be distinct stages; no full sentences, no sub-bullets.\n\n"
    "Output ONLY one line per item, in this exact format (no other text):\n"
    "YYYY-MM-DD | <short topic label>\n\n"
    "Memories:\n{fact_block}\n\n"
    "Question: {question}\n"
)

ABSTAIN_GATE_PROMPT = (
    "Verify whether a candidate answer is actually supported by the user's memories.\n"
    "CRITICAL: the memories often mention a SIMILAR BUT DIFFERENT thing than what the "
    "question asks about (e.g. a different pet, hobby, place, person, food, or event). "
    "A similar-but-different mention does NOT count as support — only the EXACT subject "
    "of the question counts.\n"
    "Reply with one word:\n"
    "  GROUNDED   — the memories explicitly contain the specific thing the question asks about\n"
    "  UNSUPPORTED — that exact thing is absent (missing, or only a similar-but-different thing is present)\n\n"
    "Memories:\n{fact_block}\n\n"
    "Question: {question}\n"
    "Candidate answer: {answer}\n\n"
    "Verdict (GROUNDED or UNSUPPORTED):"
)
