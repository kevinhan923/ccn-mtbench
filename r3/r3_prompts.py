"""Prompt templates for the R3 normalizer LLM (RESEARCH_PLAN §R3, v3.12).

Status: **FROZEN 2026-07-29** after the 32-sentence stratified dry-run. Any edit
from here requires a RESEARCH_PLAN changelog entry and a rerun of everything
produced with the old text (no metric shopping).

What the dry-run changed, and why (the only change from v1-draft): A2 gained
DECLINE_CLAUSE, so the LLM may keep a marked span unchanged. v1-draft forced a
replacement for every detector span while A1 was free to return an empty span
list, making A2's action space strictly smaller than A1's — so A2 - A1 would
have partly measured that asymmetry instead of the value of detector
information (red line 4 requires the two be aligned except for where spans come
from). It is also the content-based stand-in for the confidence-based
abstention section R3 specifies, which the fused detector cannot support
because its per-channel confidence scales are not comparable.

Measured on the dry-run, NOT a justification: the clause changed 6 of 32
sentences and exact restoration 7/32 -> 8/32 (within noise at n=32; no claim is
made that it helps). It is safe: of 22 gold spans none was wrongly declined,
and of 16 non-gold spans 2 were declined. Artifacts:
r3_exp_result/normalized/dryrun{,_decline}/.

All three tiers share one skeleton, one category glossary, and one output
format, so A2's only extra information over A1 is the detector's spans and
candidates, and A3's only extra over A2 is that its spans are gold
(RESEARCH_PLAN: 候选只能来自"找噪声的那一方"; A3 gets gold span+category but
never noise_std). Example pairs below are verified absent from BOTH
contrastive tables on both the noisy and std side (no gold leakage into the
prompt): 童鞋/同学, bhys/不好意思, 干饭/吃饭, hold住/稳住.
"""

PROMPTS_VERSION = "r3-prompts-v2-frozen-20260729"

GLOSSARY = """\
Noise categories:
- HOM: homophone or near-homophone respelling of a standard word (e.g. 童鞋 for 同学)
- PYA: pinyin initialism typed as Latin letters (e.g. bhys for 不好意思)
- NEO: internet neologism or slang standing for a plain expression (e.g. 干饭 for 吃饭)
- MIX: Latin/English word inserted in Chinese text (e.g. hold住 for 稳住)"""

RULES = """\
Rules:
- "std" must be standard written Chinese and a drop-in replacement for exactly the span text; everything outside the spans stays unchanged.
- Preserve the sentence's meaning, tone and register. Do not translate, do not explain, do not add words outside the span.
Output ONLY a JSON object, nothing else:
{"spans": [{"text": "<span text>", "type": "<HOM|PYA|NEO|MIX>", "std": "<replacement>"}]}"""

_HEAD = ("You are normalizing noisy Chinese social-media text so it can be "
         "machine-translated. ")


def _span_lines(spans: list[dict], with_candidates: bool) -> str:
    lines = []
    for i, sp in enumerate(spans, 1):
        line = f"{i}. text=\"{sp['text']}\" type={sp['type']}"
        if with_candidates and sp.get("candidates"):
            line += " candidates=[" + ", ".join(sp["candidates"]) + "]"
        lines.append(line)
    return "\n".join(lines)


def build_a1_prompt(src: str) -> str:
    """A1: the LLM finds the noise itself, then normalizes it."""
    return (_HEAD + "Identify every noisy span in the sentence and give its "
            "standard replacement.\n\n" + GLOSSARY + "\n\n"
            f"Sentence: {src}\n\n"
            "For a HOM span, consider pronunciation-based replacement "
            "candidates before choosing \"std\".\n"
            "If the sentence contains no such noise, output {\"spans\": []}.\n"
            + RULES)


DECLINE_CLAUSE = ("The detector is not perfect and may mark text that is not "
                  "actually noise. If a marked span is already standard Chinese, "
                  "set its \"std\" to the span text unchanged.\n")


def build_a2_prompt(src: str, spans: list[dict], allow_decline: bool = True) -> str:
    """A2: detector-marked spans (type + optional candidates from the detector).

    allow_decline=True is the FROZEN contract (see module docstring): the LLM
    may keep a marked span unchanged by echoing its text as "std". Pass False
    only to reproduce the superseded v1-draft forced-replacement variant as an
    ablation. Detector RECALL is unaffected either way — a span the detector
    never emitted stays unrepairable in both.
    """
    return (_HEAD + "A noise detector marked the following span(s). For EACH "
            "marked span, give its standard replacement.\n\n" + GLOSSARY + "\n\n"
            f"Sentence: {src}\n"
            f"Marked spans:\n{_span_lines(spans, with_candidates=True)}\n\n"
            "If a candidates list is given, prefer the best candidate unless "
            "every candidate is wrong in this context.\n"
            + (DECLINE_CLAUSE if allow_decline else "")
            + "Echo the spans in the given order with the given text and type, "
            "adding \"std\" to each.\n" + RULES)


def build_a3_prompt(src: str, spans: list[dict]) -> str:
    """A3: gold spans and gold categories, no candidates, no gold std."""
    return (_HEAD + "The noisy span(s) in this sentence are given, with their "
            "categories. For EACH span, give its standard replacement.\n\n"
            + GLOSSARY + "\n\n"
            f"Sentence: {src}\n"
            f"Noisy spans:\n{_span_lines(spans, with_candidates=False)}\n\n"
            "Echo the spans in the given order with the given text and type, "
            "adding \"std\" to each.\n" + RULES)
