#!/usr/bin/env python3
"""Pre-flight for the R3b2 arms — runs with no model, no GPU, no network.

Gates, in order of how badly they would corrupt the result if broken:
  1. GOLD LEAKAGE. Every glossary example must occur zero times in either
     contrastive table. This is the check whose prose version in v1 was wrong:
     v1's 童鞋/同学 pair is exactly the span and standard form of B-HOM-003 and
     B-HOM-004, so every v1 arm carrying the glossary saw those two answers.
  2. LADDER DISCIPLINE. b1s/b2s/b3s must share one head and one tail verbatim,
     so b2s-b1s is the hint block and nothing else.
  3. OUTPUT CONTRACT. Present once, identical in all three arms, and the prompt
     must end on the generation cue rather than on v1's echo-prone
     "Output only the translated result: {src}".
  4. VIOLATION DETECTOR. Fires on the real v1 failure strings and stays silent
     on real v1 successes — otherwise the mechanism table means nothing.

    python3 selftest_r3b2.py        # from r3/; exit 0 = safe to ship to the pod
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "r1-r2")))

import r3b_prompts as v1
import r3b_prompts_v2 as v2
from compare_r3b import contract_violation
from lib_data import load_contrastive

TABLES = [os.path.join(HERE, "..", "r1-r2", "data", f"contrastive_{t}.tsv")
          for t in ("A", "B")]

_fails: list[str] = []


def chk(cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


def table_rows() -> list[tuple[str, str]]:
    """(uid, every text field joined) for all rows of both tables."""
    out = []
    for path in TABLES:
        for r in load_contrastive(path):
            out.append((r["uid"], " ||| ".join(
                [r["src_noisy"], r["src_clean"], r["ref_en"]]
                + list(r["noise_span"]) + list(r["gold_en"]))))
    return out


def main() -> int:
    rows = table_rows()
    print(f"[0] audit surface: {len(rows)} rows across {len(TABLES)} tables")
    chk(len(rows) == 759, "759 rows loaded (267 + 492)")

    print("[1] gold leakage — glossary examples must not exist in the data")
    for noisy, std in v2.GLOSSARY_EXAMPLES:
        for form, kind in ((noisy, "noisy form"), (std, "standard form")):
            hits = [uid for uid, blob in rows if form in blob]
            chk(not hits, f"{kind} {form!r}: {len(hits)} occurrence(s) "
                          f"{hits[:4] if hits else ''}")
        chk(noisy in v2._GLOSSARY and std in v2._GLOSSARY,
            f"pair {noisy}/{std} is the text actually shown in the prompt")
    # the v1 leak this gate exists to catch must still be detectable
    v1_leak = [uid for uid, blob in rows if "童鞋" in blob]
    chk(len(v1_leak) == 2 and v1_leak == ["B-HOM-003", "B-HOM-004"],
        f"regression witness: v1's 童鞋 example still leaks on {v1_leak}")
    chk("童鞋" not in v2._GLOSSARY, "v2 no longer ships the leaking example")

    print("[2] ladder discipline")
    src = "测试句子"
    p1 = v2.build_b1_prompt(src)
    p2 = v2.build_b2_prompt(src, [{"text": "菌", "type": "HOM",
                                   "candidates": ["俊"], "confidence": 0.9}])
    p3 = v2.build_b3_prompt(src, [{"text": "菌", "type": "HOM", "std": "俊"}])
    head = v2._HEAD
    chk(all(p.startswith(head) for p in (p1, p2, p3)), "one shared head")
    tail = v2._TAIL.format(src=src)
    chk(all(p.endswith(tail) for p in (p1, p2, p3)), "one shared tail, verbatim")
    chk(p2 == head + "\n\n" + p2[len(head) + 2:-len(tail)] + tail,
        "b2s = head + hint block + tail, nothing else")
    chk("Detector hints:" in p2 and "Detector hints:" not in p1,
        "the hint block is the only thing b2s adds over b1s")

    print("[3] output contract")
    for name, p in (("b1s", p1), ("b2s", p2), ("b3s", p3)):
        chk(p.count("Output rules") == 1, f"{name}: contract present exactly once")
        chk(p.rstrip().endswith("English translation:"),
            f"{name}: ends on the generation cue")
        chk("Output only the translated result" not in p,
            f"{name}: v1's echo-prone tail is gone")
    for clause in ("single line", "Do not explain", "HOM, PYA, NEO or MIX",
                   "Do not repeat the Chinese sentence", "quotation marks"):
        chk(all(clause in p for p in (p1, p2, p3)), f"clause in all arms: {clause!r}")
    chk("imperfect" in p2 and "disregard" in p2,
        "b2s keeps hints fallible and refusable")
    chk("do not report that decision" in p2, "b2s tells the refusal to be silent")
    chk("(the detector marked nothing in this sentence)"
        in v2.build_b2_prompt(src, []), "empty-detector case keeps the block")

    print("[4] violation detector vs REAL v1 rows")
    real_src = "谨以此片献给每一位纯爱战神"
    cases = [
        ('The sentence with noise resolved is: "x"\n\nTranslation: "y"', True, "A-MIX-002 shape"),
        ('Zhangzhou local NEO "蹲蹲" means eager', True, "A-NEO-028 code leak"),
        ('Translate the sentence into English so that the meaning the author '
         'intended is preserved, resolving any such noise to its intended sense. '
         'Output only the translated result: x', True, "A-NEO-180 instruction echo"),
        (real_src + "\n\nTranslation: dedicating this film", True, "B-NEO-018 source echo"),
        ('Prince Guang, I am also.', False, "clean b0 output"),
        ('Super mean old ladies are usually English teachers', False, "clean b0 output"),
        ('Kuala Lumpur\'s ultimate must-avoid points!', False, "clean b0 output"),
    ]
    for hyp, want_bad, label in cases:
        got = contract_violation(hyp, real_src)
        chk(bool(got) == want_bad, f"{label}: violation={got}")

    print(f"\nversions: v1={v1.R3B_PROMPTS_VERSION}  v2={v2.R3B_PROMPTS_VERSION}")
    print(f"FAILURES: {len(_fails)}")
    for f in _fails:
        print(f"  - {f}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
