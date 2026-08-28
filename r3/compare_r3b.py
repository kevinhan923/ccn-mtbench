#!/usr/bin/env python3
"""R3b verdict: paired bootstrap on noisy-side scores across the four arms.

Because every arm shares one clean side (copied from Run 4's qwen3-8b row) and
one scoring pass, comparing noisy-side scores IS comparing deltas — the clean
term cancels. Higher noisy-side score = better, so diff > 0 means the first
arm wins.

Pre-registered gate (RESEARCH_PLAN §R3b v3.17a, locked before any number existed):
  PRIMARY    b2 - b0  CI lower bound > 0   (detector+model beats the model's own
                                            benchmark row — a PIPELINE-level claim;
                                            paper wording must not attribute the
                                            gain to detector information alone)

POST-HOC arms (RESEARCH_PLAN §R3b2, added 2026-07-30 AFTER the v1 gate failed):
b1s/b2s/b3s rerun the ladder with the v2 prompts, which add a hard output
contract. Every number involving them is post-hoc by construction and must be
labelled as such — the v1 b2-b0 line above remains the pre-registered result.
The point of the ladder is that b2s-b1s holds the prompt scaffold constant and
so isolates DETECTOR INFORMATION, which no contrast in the v1 data can do.

The contract-violation table published below is the mechanism check: it counts,
per arm, the rows whose output stopped being a bare translation (multi-line,
meta-commentary, category codes leaking through, source echoed back). v1's b2
had 10/267 and 10/492 such rows averaging -29 / -36 XCOMET, which is where half
its deficit came from. If the repair worked, b2s's count is near b0's.

    python3 compare_r3b.py                       # default result dirs
    python3 compare_r3b.py --resdir ../r3_exp_result/r3b/results_A --table A
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_R1R2 = os.path.normpath(os.path.join(HERE, "..", "r1-r2"))
if _R1R2 not in sys.path:
    sys.path.insert(0, _R1R2)

from lib_analysis import load_segments, paired_bootstrap_diff
from lib_data import CATEGORIES

MODEL = "qwen3-8b"
# (label, filename stem). b0 is the copied Run 4 row, scored in the same pass.
ARM_FILES = [("b0", MODEL), ("b1", f"{MODEL}-b1"),
             ("b2", f"{MODEL}-b2"), ("b3", f"{MODEL}-b3"),
             ("b1s", f"{MODEL}-b1s"), ("b2s", f"{MODEL}-b2s"),
             ("b3s", f"{MODEL}-b3s")]
PAIRS = [("b2", "b0", "PRIMARY gate (pre-registered, v1 prompts)"),
         ("b2", "b1", "exploratory: detector info alone"),
         ("b1", "b0", "exploratory: awareness alone"),
         ("b3", "b2", "exploratory: detector headroom"),
         ("b2s", "b1s", "POST-HOC KEY: detector info, scaffold held constant"),
         ("b2s", "b0", "POST-HOC: repaired pipeline vs benchmark"),
         ("b1s", "b0", "POST-HOC: cost/benefit of the scaffold alone"),
         ("b3s", "b2s", "POST-HOC: headroom left to detector quality"),
         ("b3s", "b1s", "POST-HOC: oracle ceiling over the control"),
         # Added 2026-07-30 after the first verdict: the oracle-vs-benchmark row
         # was missing from this list, so the ladder's headline comparison had a
         # point estimate but no CI. Adding the pair only computes the statistic
         # for a contrast that was already part of the post-hoc design; no arm,
         # protocol or metric changed.
         ("b3s", "b0", "POST-HOC: oracle vs the benchmark row itself"),
         ("b2s", "b2", "POST-HOC: what the output contract alone bought")]

# Mechanism check. Deliberately mechanical and fixed before the v2 run so it
# cannot be tuned to flatter an arm: any of these means the row stopped being a
# bare one-line translation. Derived from the v1 failures actually observed.
_META = re.compile(
    r"(Translation:|translation of the sentence|Note:|Resolving the noise|"
    r"the sentence translates to|The sentence with noise resolved|"
    r"Output only the translated result|resolving any such noise|"
    r"is a pinyin initialism|is a homophone for|internet slang|"
    r"more natural translation|Final translation|is an internet neologism)")
_CODES = re.compile(r"\b(HOM|PYA|NEO|MIX)\b")


def contract_violation(hyp: str, src: str) -> str | None:
    """Which contract a row broke, or None. First match wins, fixed order."""
    if not hyp or not hyp.strip():
        return "empty"
    if "\n" in hyp.strip():
        return "multiline"
    if _META.search(hyp):
        return "meta"
    if _CODES.search(hyp):
        return "code"
    if src.strip() and src.strip() in hyp:
        return "echo_src"
    return None


def load_arms(resdir):
    arms = {}
    for label, stem in ARM_FILES:
        path = os.path.join(resdir, f"segments_{stem}.tsv")
        if not os.path.exists(path):
            print(f"  [missing] {label}: {path}")
            continue
        arms[label] = load_segments(path)
    return arms


def aligned(recs_a, recs_b, field, cat=None):
    by_a = {r["uid"]: r[field] for r in recs_a
            if r.get(field) is not None and (cat is None or r["category"] == cat)}
    by_b = {r["uid"]: r[field] for r in recs_b if r.get(field) is not None}
    uids = [u for u in by_a if u in by_b]
    return [by_a[u] for u in uids], [by_b[u] for u in uids]


def fmt(res, tag=""):
    if res["n"] == 0:
        return "n=0"
    gate = "PASS" if res["lo"] > 0 else "fail"
    return (f"{res['diff']:+7.2f}  [{res['lo']:+6.2f}, {res['hi']:+6.2f}]  "
            f"p={res['p_boot']:.4f}  n={res['n']:3d}  gate {gate}  {tag}")


def report(resdir, table):
    print(f"\n=== table {table}  ({resdir}) ===")
    arms = load_arms(resdir)
    if len(arms) < 2:
        print("  not enough arms scored yet")
        return

    # clean-side identity: every arm copied b0's clean bytes, so clean scores
    # from the shared scoring pass must agree — this is the reuse self-proof.
    if "b0" in arms:
        base = {r["uid"]: r["xcomet_clean"] for r in arms["b0"]}
        for label, _stem in ARM_FILES:
            if label == "b0" or label not in arms:
                continue
            bad = sum(1 for r in arms[label]
                      if r.get("xcomet_clean") is not None
                      and base.get(r["uid"]) is not None
                      and abs(r["xcomet_clean"] - base[r["uid"]]) > 1e-6)
            flag = "identical (reuse verified)" if bad == 0 else f"*** {bad} rows differ ***"
            print(f"  clean-side {label} vs b0: {flag}")

    # mechanism check: how often each arm broke the one-line contract, and what
    # those rows cost. A near-zero count for b2s is the evidence the repair
    # targeted the right thing; a still-high count means it did not.
    print("\n  [output-contract violations, noisy side]")
    b0_by_uid = {r["uid"]: r for r in arms.get("b0", [])}
    for label, _stem in ARM_FILES:
        if label not in arms:
            continue
        recs = arms[label]
        bad = [(r, contract_violation(r["hyp_noisy"], r["src_noisy"])) for r in recs]
        bad = [(r, why) for r, why in bad if why]
        kinds = {}
        for _r, why in bad:
            kinds[why] = kinds.get(why, 0) + 1
        cost = ""
        if bad and label != "b0" and b0_by_uid:
            deltas = [r["xcomet_noisy"] - b0_by_uid[r["uid"]]["xcomet_noisy"]
                      for r, _ in bad
                      if r["uid"] in b0_by_uid
                      and r["xcomet_noisy"] is not None
                      and b0_by_uid[r["uid"]]["xcomet_noisy"] is not None]
            if deltas:
                mean_all = sum(deltas) / len(recs)
                cost = (f"  mean XCOMET vs b0 on them {sum(deltas)/len(deltas):+6.2f}"
                        f"  (pulls the arm's overall mean by {mean_all:+.2f})")
        print(f"    {label:4s} {len(bad):3d}/{len(recs)} rows  {kinds if kinds else ''}{cost}")

    for field, name in (("xcomet_noisy", "XCOMET"), ("nta_noisy", "NTA")):
        print(f"\n  [{name}, noisy side]")
        for hi_arm, lo_arm, tag in PAIRS:
            if hi_arm not in arms or lo_arm not in arms:
                continue
            a, b = aligned(arms[hi_arm], arms[lo_arm], field)
            print(f"    {hi_arm}-{lo_arm}: {fmt(paired_bootstrap_diff(a, b), tag)}")

    # ("b3s","b0") added 2026-07-30 by the story-A audit: RESULTS_R3B2.md quotes the
    # oracle's per-category effect but NO machine artifact held it, so its CIs could
    # not be cited. The paper's strongest claim — table B's aggregate null is a
    # cancellation between the categories noise actually damages (HOM, NEO) and the
    # two our pre-registration predicted would lose (PYA, MIX, the PheMT
    # redundant-expansion mechanism) — needs those CIs. Emitting the pair changes no
    # protocol, arm, metric or seed; it only computes a contrast already published.
    for hi_arm, lo_arm in (("b2", "b1"), ("b2s", "b1s"), ("b2s", "b0"),
                           ("b3s", "b0"), ("b3s", "b1s")):
        if hi_arm not in arms or lo_arm not in arms:
            continue
        for field, name in (("xcomet_noisy", "XCOMET"), ("nta_noisy", "NTA")):
            print(f"\n  [{name} noisy, {hi_arm}-{lo_arm} by category — exploratory]")
            for cat in CATEGORIES:
                a, b = aligned(arms[hi_arm], arms[lo_arm], field, cat=cat)
                if a:
                    print(f"    {cat}: {fmt(paired_bootstrap_diff(a, b))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../r3_exp_result/r3b",
                    help="r3b root holding results_A/ and results_B/")
    ap.add_argument("--tag", default="",
                    help="read results_<table><tag>/ instead — the R3b2 rerun "
                         "scores into results_A_r3b2/ so v1's frozen segments "
                         "files are never overwritten")
    ap.add_argument("--table", choices=("A", "B"), help="one table only")
    args = ap.parse_args()

    tables = [args.table] if args.table else ["A", "B"]
    for tb in tables:
        report(os.path.join(args.base, f"results_{tb}{args.tag}"), tb)


if __name__ == "__main__":
    main()
