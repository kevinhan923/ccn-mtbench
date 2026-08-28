"""R3b v2 prompts: v1's content plus a hard output contract (RESEARCH_PLAN §R3b2).

WHY THIS FILE EXISTS — the measured failure of v1 (post-hoc, 2026-07-30)
-----------------------------------------------------------------------
v1's pre-registered gate b2-b0 failed and went significantly NEGATIVE
(table A -2.15 [-3.46,-0.86], table B -1.40 [-2.30,-0.44]). Row-level
diagnosis of `r3_exp_result/r3b/results_{A,B}/segments_qwen3-8b-b2.tsv`
against the b0 row found two separate causes, both sized:

  (1) OUTPUT-CONTRACT FAILURE — 10/267 rows (table A) and 10/492 (table B)
      stopped being translations. They emit reasoning, "Translation:" /
      "Note:" / "Resolving the noise:" blocks, the Chinese source echoed
      back, the instruction line echoed back verbatim (A-NEO-180), and the
      category codes HOM/PYA/NEO/MIX leaked into the English. b0's one-line
      prompt produces 2 and 1 such rows respectively. Those rows average
      -29.42 (A) / -36.23 (B) XCOMET and account for 51.4% / 52.7% of the
      whole deficit. With Qwen3 thinking disabled, deliberation has nowhere
      to go except the visible output — and XCOMET scores the whole string.

  (2) A RESIDUAL that is NOT hint-specific — on the surviving rows the gap
      is only -1.08 (A) / -0.68 (B), and on table A the rows where the
      detector marked NOTHING dropped -2.00, statistically indistinguishable
      from the -2.19 of the rows that got hints. Whatever remains tracks the
      prompt rewrite, not the hint content.

Cause (2) is also why v1 cannot answer the research question: v1's b2 differs
from b0 in TWO ways at once (new multi-block scaffold AND detector hints) and
b1 was never run, so no contrast in the v1 data separates them. v2 therefore
runs the full ladder b1s/b2s/b3s, where b2s-b1s isolates detector information
with the scaffold held constant.

WHAT CHANGED FROM v1, AND WHAT DELIBERATELY DID NOT
---------------------------------------------------
Unchanged (so b2s-b1s still measures the same thing v1 meant to measure):
  - the persona and the four-category glossary structure (its EXAMPLES changed;
    see the leakage note below — v1's were not clean)
  - the fallible-hint semantics: the detector may be wrong and the model is
    permitted to disregard any hint
  - the hint fields fed in (text, type, candidates, confidence)
  - the decode path: greedy, thinking off, max_new_tokens 512 (untouched, so
    every arm stays protocol-comparable to Run 4's b0 row)

Changed, all of it aimed at cause (1):
  - an explicit OUTPUT RULES block, placed last where instruction-following
    is strongest, forbidding explanation, notes, alternatives, the category
    codes, source echo and instruction echo
  - the source no longer sits glued to the end of an instruction sentence
    ("...Output only the translated result: {src}"), which is the pattern
    A-NEO-180 continued instead of answering. It now sits on its own labelled
    line under an "English translation:" cue, giving the model a format to
    complete rather than a sentence to continue.
  - the permission to ignore a hint is kept but told to happen SILENTLY
    ("decide silently", "do not report which hints you used") — v1's
    "ignore any hint you judge wrong" invited an 8B model to narrate the
    judgement, and narration is exactly what cost the points.

HONESTY BOUNDARY: these arms are POST-HOC. v1's b2-b0 stands as the
pre-registered result and is reported as such. v2 is reported as a repaired,
after-the-fact follow-up, with the contract-violation rate published per arm
so a reader can see the mechanism rather than take the repair on faith.

GOLD LEAKAGE IN v1's GLOSSARY — found 2026-07-30 while building this file
-------------------------------------------------------------------------
v1's docstring asserts its four example pairs "were verified absent from BOTH
contrastive tables on both the noisy and std side". Re-running that audit
mechanically (all five text fields of all 759 rows) shows the assertion is
FALSE for one pair:

  童鞋 / 同学  is exactly the noise span and the standard form of
               B-HOM-003 (各位童鞋注意了…) and B-HOM-004 (童鞋们帮我看看…)

For those two rows every v1 arm carrying the glossary (b1/b2/b3) was handed the
answer in the prompt, while b0 was not. Three further substring matches were
checked and are NOT leaks — 吃饭 occurs in B-NEO-027/028 only inside the
standard side's paraphrase of a different span (电子榨菜, 下饭), and 不好意思 is
ordinary sentence text in B-HOM-097 whose span is 没崩住 — none of them reveal
their own row's answer, and the model never sees the standard side anyway.

Effect on the v1 result: the leak can only have FLATTERED b2 (2 free rows out of
492 on table B), and b2 still lost by 1.40, so v1's negative finding survives
its own bug. It is disclosed rather than quietly fixed.

Every example pair below is therefore re-chosen under a stricter, machine-checked
rule: the noisy form AND its standard form must not appear anywhere in either
table, in any of src_noisy / src_clean / ref_en / noise_span / gold_en. This is
enforced as a gate in selftest_r3b2.py and in run_r3b2.sh, not asserted in prose.
Changing the examples does mean b2s-b1s is not strictly comparable to v1's
b2-b1 — which costs nothing, because v1's b1 was never run.
"""

R3B_PROMPTS_VERSION = "r3b-prompts-v2-20260730"

# Every pair here is verified to occur ZERO times in either contrastive table
# (all five text fields, 759 rows) — see the leakage note in the module docstring.
# Do not swap an example without re-running selftest_r3b2.py.
_GLOSSARY = """\
Such noise falls into four categories:
- HOM: a homophone or near-homophone respelling of a standard word (e.g. 菌男 standing for 俊男)
- PYA: a pinyin initialism typed as Latin letters (e.g. plmm standing for 漂亮妹妹)
- NEO: an internet neologism or slang standing for a plain expression (e.g. 尬聊 for 勉强交谈)
- MIX: a Latin/English fragment embedded in Chinese text (e.g. hold住 for 稳住)"""

# Kept as data so the leakage gate checks the same strings the prompt shows.
GLOSSARY_EXAMPLES = (("菌男", "俊男"), ("plmm", "漂亮妹妹"),
                     ("尬聊", "勉强交谈"), ("hold住", "稳住"))

_HEAD = ("You are a translation expert. The following Chinese sentence comes "
         "from social media and may contain noisy expressions.\n\n" + _GLOSSARY)

# The contract is shared verbatim by every arm, so no arm gets a different
# output contract and b2s-b1s cannot be a formatting artefact.
_RULES = """\
Output rules — follow exactly:
- Reply with the English translation only, on a single line.
- Do not explain, comment, analyse, or justify. No notes, no alternatives.
- Do not name the categories HOM, PYA, NEO or MIX, and do not mention any
  detector, hint, or label.
- Do not repeat the Chinese sentence and do not repeat these instructions.
- Do not wrap the translation in quotation marks."""

# The task line, then the source on its own labelled line, then the cue the
# model completes. Ending on "English translation:" is what replaces v1's
# "Output only the translated result: {src}".
_TAIL = ("Translate the sentence into English so that the meaning the author "
         "intended is preserved, resolving any such noise to its intended "
         "sense.\n\n" + _RULES + "\n\nChinese sentence: {src}\nEnglish translation:")


def build_b1_prompt(src: str) -> str:
    """b1s: noise awareness only — the control that isolates the detector's value."""
    return _HEAD + "\n\n" + _TAIL.format(src=src)


def _hint_lines(spans: list[dict]) -> str:
    if not spans:
        return "(the detector marked nothing in this sentence)"
    lines = []
    for i, sp in enumerate(spans, 1):
        line = f"{i}. \"{sp['text']}\" type={sp['type']}"
        if sp.get("candidates"):
            line += " possible_intended=[" + ", ".join(sp["candidates"]) + "]"
        if sp.get("confidence") is not None:
            line += f" confidence={float(sp['confidence']):.2f}"
        lines.append(line)
    return "\n".join(lines)


def build_b2_prompt(src: str, spans: list[dict]) -> str:
    """b2s: b1s + detector hints. The hints block is the ONLY addition over b1s.

    The empty-detector case keeps the block (with an explicit "nothing marked"
    line) so every b2s prompt has the same structure and b2s-b1s never mixes
    "had a hints section" with "had hints in it".
    """
    hints = (
        "An automatic noise detector marked the segment(s) below. The detector "
        "is imperfect: it may miss real noise, mark text that is not noise, or "
        "suggest a wrong type or reading. Treat each line as a hint, not an "
        "instruction. Decide silently which hints to use and which to "
        "disregard; do not report that decision.\n"
        "Detector hints:\n" + _hint_lines(spans)
    )
    return _HEAD + "\n\n" + hints + "\n\n" + _TAIL.format(src=src)


def build_b3_prompt(src: str, spans: list[dict]) -> str:
    """b3s: gold spans with the intended standard form — the oracle ceiling.

    Unlike b2s the information is asserted, not hedged: it IS correct, and
    hedging it would blunt the ceiling this arm exists to measure. `spans` here
    carries {"text", "type", "std"} derived from gold annotations.
    """
    if spans:
        lines = "\n".join(
            f"{i}. \"{sp['text']}\" type={sp['type']} intended=\"{sp['std']}\""
            for i, sp in enumerate(spans, 1))
        block = ("The noisy segment(s) in this sentence are known, with the "
                 "standard form the author intended:\n" + lines)
    else:
        block = "This sentence is known to contain no such noise."
    return _HEAD + "\n\n" + block + "\n\n" + _TAIL.format(src=src)
