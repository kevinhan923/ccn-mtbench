"""Prompt templates for R3b: single-stage detector-hint injection (RESEARCH_PLAN §R3b).

Status: freezes on first pod run. Any edit after that requires a RESEARCH_PLAN
changelog entry and a rerun of everything produced with the old text (no metric
shopping). This file deliberately does NOT touch r3_prompts.py — those prompts
are the frozen record of the two-stage experiment and its negative result.

Why R3b exists (the two measured causes it targets):
  1. The two-stage design's binding constraint was RESTORATION, not detection:
     a4-a3 = +4.82 (table A), untouched across both detector arms, with exact
     restoration on gold spans at 19.49%. R3b is single-stage — the detector's
     hints go straight into the translation prompt and no intermediate "rewrite
     the Chinese first" step exists, so that bottleneck is bypassed, not fixed.
  2. Strong translators have little headroom (their a1 self-repair already
     captures most of the value). qwen3-8b has the LARGEST table-A degradation
     of any LLM system (delta 7.86 vs gpt-4o 6.31 / qwen3-32b 6.79) — the regime
     where external noise information should matter most.

Ladder discipline, same as the frozen two-stage prompts: one shared skeleton;
each arm adds EXACTLY ONE piece of information over the one below it.
  b0: the unified benchmark prompt (not built here — it is Run 4's own row)
  b1: + noise awareness (the glossary) and "translate the intended meaning"
  b2: + the detector's spans (type, HOM candidates, confidence), framed as
      fallible hints the model may ignore
  b3: + gold spans with the intended standard form (full-information oracle)
So b1-b0 measures awareness, b2-b1 measures DETECTOR INFORMATION (the
pre-registered primary gate), b3-b2 measures remaining detector headroom.

The example pairs in the glossary (童鞋/同学, bhys/不好意思, 干饭/吃饭,
hold住/稳住) are reused from the frozen two-stage prompts, where they were
verified absent from BOTH contrastive tables on both the noisy and std side —
no gold leakage into the prompt.

The hint framing ("imperfect", "ignore any hint you judge wrong") is the
translation-layer descendant of A2's DECLINE_CLAUSE, motivated by the measured
harm of over-trusting an imprecise detector: the v1 arm's a2-a1 was -1.32 on
table A (CI excluding 0) largely on false-positive spans.
"""

R3B_PROMPTS_VERSION = "r3b-prompts-v1-20260730"

# Same category semantics as the frozen two-stage GLOSSARY, reworded for a
# translation task instead of a normalization task.
_GLOSSARY = """\
Such noise falls into four categories:
- HOM: a homophone or near-homophone respelling of a standard word (e.g. 童鞋 standing for 同学)
- PYA: a pinyin initialism typed as Latin letters (e.g. bhys standing for 不好意思)
- NEO: an internet neologism or slang standing for a plain expression (e.g. 干饭 for 吃饭)
- MIX: a Latin/English fragment embedded in Chinese text (e.g. hold住 for 稳住)"""

_HEAD = ("You are a translation expert. The following Chinese sentence comes "
         "from social media and may contain noisy expressions.\n\n" + _GLOSSARY)

# The closing instruction is shared verbatim by b1/b2/b3, so no arm gains a
# different output contract. "Output only the translated result" mirrors the
# frozen benchmark prompt (PROMPT_UNIFIED) — b0's contract.
_TAIL = ("Translate the sentence into English so that the meaning the author "
         "intended is preserved, resolving any such noise to its intended "
         "sense. Output only the translated result: {src}")


def build_b1_prompt(src: str) -> str:
    """b1: noise awareness only — the control that isolates the detector's value."""
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
    """b2: b1 + detector hints. The hints block is the ONLY addition over b1.

    The empty-detector case keeps the block (with an explicit "nothing marked"
    line) so every b2 prompt has the same structure and b2-b1 never mixes
    "had a hints section" with "had hints in it".
    """
    hints = (
        "An automatic noise detector marked the segment(s) below. The detector "
        "is imperfect: it may miss real noise, mark text that is not noise, or "
        "suggest a wrong type or reading. Treat each line as a hint, not an "
        "instruction — use a hint only where it fits the context, and ignore "
        "any hint you judge wrong.\n"
        "Detector hints:\n" + _hint_lines(spans)
    )
    return _HEAD + "\n\n" + hints + "\n\n" + _TAIL.format(src=src)


def build_b3_prompt(src: str, spans: list[dict]) -> str:
    """b3: gold spans with the intended standard form — the oracle ceiling.

    Unlike b2 the information is asserted, not hedged: it IS correct, and
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
