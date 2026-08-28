"""Local self-test for the §7 analysis layer (no models, no GPU).

Covers edit_distance, paired_bootstrap_diff, regression, disagreement_flags,
contamination_deltas, and the full run_analysis.py end-to-end on synthetic
segments. Deterministic (fixed seeds).  Run:  python selftest_analysis.py
"""
import csv
import os
import random
import subprocess
import sys
import tempfile

import lib_analysis as A

HERE = os.path.dirname(os.path.abspath(__file__))
_n = 0

SEG_HEADER = ["uid", "category", "xcomet_noisy", "xcomet_clean", "xcomet_d",
              "kiwi_noisy", "kiwi_clean", "kiwi_d", "nta_noisy", "nta_clean", "nta_d",
              "src_noisy", "hyp_noisy", "src_clean", "hyp_clean", "ref_en"]


def check(cond, msg):
    global _n
    _n += 1
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ok: {msg}")


def test_edit_distance():
    print("[edit_distance]")
    check(A.edit_distance("abc", "abc") == 0, "identical -> 0")
    check(A.edit_distance("", "abc") == 3, "empty vs abc -> 3")
    check(A.edit_distance("绷不住了", "蚌埠住了") == 2, "homophone twin edits 2 chars")
    check(A.edit_distance("牛马", "打工人") == 3, "neologism gloss edits more")


def test_paired_bootstrap_diff():
    print("[paired_bootstrap_diff]")
    r = A.paired_bootstrap_diff([10.0] * 30, [2.0] * 30)
    check(r["diff"] == 8.0 and r["lo"] == 8.0 and r["hi"] == 8.0, "constant separation -> diff 8, tight CI")
    check(r["p_boot"] == 0.0, "clear separation -> p_boot 0")
    r2 = A.paired_bootstrap_diff([5.0] * 20, [5.0] * 20)
    check(r2["diff"] == 0.0 and r2["p_boot"] == 1.0, "identical -> diff 0, p_boot 1")
    # varied but clearly separated -> CI excludes 0
    a = [8.0, 12.0] * 15
    b = [0.0, 4.0] * 15
    r3 = A.paired_bootstrap_diff(a, b)
    check(r3["diff"] == 8.0 and r3["lo"] > 0, "varied separation -> diff 8, CI excludes 0")
    try:
        A.paired_bootstrap_diff([1.0], [1.0, 2.0])
        check(False, "should reject length mismatch")
    except ValueError:
        check(True, "rejects length mismatch")


def test_regression():
    print("[regression]")
    rng = random.Random(0)
    cats = ["HOM", "PYA", "NEO", "MIX"]
    rows = []
    for i in range(120):
        cat = cats[i % 4]
        ed = {"HOM": 1, "PYA": 2, "NEO": 5, "MIX": 3}[cat]
        length = 20 + rng.randint(-3, 3)
        # Δ increases with edit distance (slope +2) + small noise; baseline HOM
        delta = 5.0 + 2.0 * ed + rng.gauss(0, 0.5)
        rows.append(dict(delta=delta, category=cat, length=length, edit_distance=ed))
    res = A.regression(rows, seed=1)
    names = {c["name"]: c for c in res["coefs"]}
    check(res["baseline"] == "HOM", "baseline is first present category HOM")
    check("editdist_z" in names, "regression has editdist_z term")
    check(names["editdist_z"]["estimate"] > 0 and names["editdist_z"]["excludes_zero"],
          "positive edit-distance dependence recovered, CI excludes 0")
    check(A.regression(rows[:3])["coefs"] == [], "too few rows -> no coefs (graceful)")


def test_disagreement_flags():
    print("[disagreement_flags]")
    recs = [
        dict(uid="d1", category="NEO", xcomet_d=5.0, kiwi_d=30.0, xcomet_noisy=60.0, nta_noisy=100.0,
             src_noisy="s", hyp_noisy="h"),  # xcomet/kiwi disagree by 25
        dict(uid="d2", category="HOM", xcomet_d=4.0, kiwi_d=5.0, xcomet_noisy=85.0, nta_noisy=0.0,
             src_noisy="s", hyp_noisy="h"),  # high xcomet but NTA miss
        dict(uid="d3", category="MIX", xcomet_d=5.0, kiwi_d=6.0, xcomet_noisy=70.0, nta_noisy=100.0,
             src_noisy="s", hyp_noisy="h"),  # clean, not flagged
    ]
    fl = A.disagreement_flags(recs)
    uids = [d["uid"] for d in fl]
    check(set(uids) == {"d1", "d2"}, "flags exactly the two problem items")
    check(uids[0] == "d1", "highest-severity (disagreement 25) sorted first")


def test_contamination():
    print("[contamination]")
    a = [dict(uid=f"a{i}", category="NEO", xcomet_d=10.0) for i in range(5)]
    b = [dict(uid=f"b{i}", category="NEO", xcomet_d=4.0) for i in range(5)]
    rows = {r["category"]: r for r in A.contamination_deltas(a, b)}
    check(rows["NEO"]["delta_a"] == 10.0 and rows["NEO"]["delta_b"] == 4.0, "per-category means")
    check(rows["NEO"]["gap"] == 6.0, "gap = Δ_A − Δ_B = 6 (A inflated)")
    check(rows["HOM"]["gap"] is None, "absent category -> gap None")


def _write_segments(path, model, seed):
    """Synthesize a plausible segments_<model>.tsv. Worse models get larger Δ."""
    rng = random.Random(seed)
    tier_delta = {"nllb-3.3b": 12.0, "qwen3-8b": 6.0, "gpt-4o": 2.0}[model]
    cats = ["HOM", "PYA", "NEO", "MIX"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(SEG_HEADER)
        for i in range(48):
            cat = cats[i % 4]
            src_noisy = "这句子测试噪声" + str(i)
            # edits vary by category so edit-distance-by-category + covariate have signal
            src_clean = src_noisy.replace("噪声", {"HOM": "燥声", "PYA": "zs",
                                                    "NEO": "背景干扰因素", "MIX": "noise"}[cat])
            xc_n = 70.0 + rng.uniform(-5, 5)
            xc_c = xc_n + tier_delta + rng.uniform(-1, 1)
            kw_n, kw_c = xc_n - 2, xc_c - 2
            nta_n = 100.0 if i % 5 else 0.0  # mostly hit; some misses to exercise flags
            w.writerow([f"u{i}", cat, f"{xc_n:.2f}", f"{xc_c:.2f}", f"{xc_c - xc_n:.2f}",
                        f"{kw_n:.2f}", f"{kw_c:.2f}", f"{kw_c - kw_n:.2f}",
                        f"{nta_n:.2f}", "100.00", f"{100.0 - nta_n:.2f}",
                        src_noisy, "hyp noisy", src_clean, "hyp clean", "ref"])


def test_end_to_end():
    print("[end-to-end run_analysis]")
    tmp = tempfile.mkdtemp()
    resA = os.path.join(tmp, "results_A")
    resB = os.path.join(tmp, "results_B")
    os.makedirs(resA); os.makedirs(resB)
    for m, s in [("nllb-3.3b", 1), ("qwen3-8b", 2), ("gpt-4o", 3)]:
        _write_segments(os.path.join(resA, f"segments_{m}.tsv"), m, s)
        _write_segments(os.path.join(resB, f"segments_{m}.tsv"), m, s + 10)

    r = subprocess.run([sys.executable, "run_analysis.py", "--resdir-a", resA,
                        "--resdir-b", resB], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"run_analysis failed:\n{r.stdout}\n{r.stderr}")

    for name in ("table_rq1_pairs.tsv", "table_rq2_regression.tsv", "table_edit_distance.tsv",
                 "table_contamination.tsv", "analysis.md"):
        check(os.path.exists(os.path.join(resA, name)), f"wrote {name}")

    with open(os.path.join(resA, "table_rq1_pairs.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    pairs = {(r["model_a"], r["model_b"]) for r in rows}
    check(("nllb-3.3b", "gpt-4o") in pairs, "rq1 has the NLLB vs GPT-4o pair")
    nllb_gpt = next(r for r in rows if (r["model_a"], r["model_b"]) == ("nllb-3.3b", "gpt-4o"))
    check(float(nllb_gpt["diff"]) > 0 and float(nllb_gpt["ci_lo"]) > 0,
          "NLLB less robust than GPT-4o (diff>0, CI excludes 0)")

    with open(os.path.join(resA, "table_edit_distance.tsv"), encoding="utf-8") as f:
        ed = {r["category"]: r for r in csv.DictReader(f, delimiter="\t")}
    check(float(ed["NEO"]["mean_editdist"]) > float(ed["HOM"]["mean_editdist"]),
          "NEO twin edits more than HOM (covariate has signal)")

    check(os.path.exists(os.path.join(resA, "flagged_nllb-3.3b.tsv")), "wrote per-model flagged file")


if __name__ == "__main__":
    test_edit_distance()
    test_paired_bootstrap_diff()
    test_regression()
    test_disagreement_flags()
    test_contamination()
    test_end_to_end()
    print(f"\nALL {_n} CHECKS PASSED")
