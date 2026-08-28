"""Local self-test for the eval pipeline (no models needed).

Covers: NTA matching logic, gold filtering, bootstrap, TSV loading/validation,
run_score aggregation (controlled hits), and the full dummy end-to-end.
Run:  python selftest.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import lib_metrics as M
from lib_data import load_contrastive
from run_score import score_file, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "data", "fixture_contrastive.tsv")
_n = 0


def check(cond, msg):
    global _n
    _n += 1
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ok: {msg}")


def test_metrics():
    print("[metrics]")
    check(M.norm_text("  HeLLo  ") == "hello", "norm_text casefold+strip")
    check(M.clean_gold(["!", "…", "a", "GOAT", " "]) == ["GOAT"], "clean_gold drops punct/short")
    check(M.nta_hit("the life of a corporate drone", ["corporate drone"]) == 1, "nta_hit positive")
    check(M.nta_hit("something unrelated entirely", ["corporate drone"]) == 0, "nta_hit negative")
    check(M.nta_hit("You are the GOAT here!", ["the goat"]) == 1, "nta_hit casefold match")
    check(M.nta_hit("anything", []) == 0, "nta_hit empty gold -> 0")
    lo, hi = M.bootstrap_ci([5.0] * 20)
    check(abs(lo - 5) < 1e-9 and abs(hi - 5) < 1e-9, "bootstrap constant -> tight CI")
    lo, hi = M.bootstrap_ci([0.0, 10.0] * 25)
    check(lo < 5 < hi, "bootstrap brackets the mean")


def test_load():
    print("[load]")
    recs = load_contrastive(FIXTURE)
    check(len(recs) == 7, "fixture has 7 rows")
    check(recs[3]["gold_en"] == ["corporate drone", "working stiff", "wage slave"],
          "gold_en split on ‖")
    check(recs[6]["gold_en"] == [], "empty gold parses to []")
    for bad, why in [
        ("uid\ttable\tcategory\tsrc_noisy\tsrc_clean\tref_en\tnoise_span\tnoise_gold_en\n"
         "x\tA\tZZZ\ta\tb\tc\td\te\n", "bad category"),
        ("uid\ttable\tcategory\tsrc_noisy\tsrc_clean\tref_en\tnoise_span\tnoise_gold_en\n"
         "x\tA\tHOM\ta\tb\tc\td\te\n"
         "x\tA\tHOM\ta\tb\tc\td\te\n", "duplicate uid"),
        ("uid\tcategory\tsrc_noisy\n" "x\tHOM\ta\n", "missing columns"),
    ]:
        p = tempfile.mktemp(suffix=".tsv")
        open(p, "w", encoding="utf-8").write(bad)
        try:
            load_contrastive(p)
            check(False, f"should reject: {why}")
        except ValueError:
            check(True, f"rejects {why}")
        finally:
            os.unlink(p)


def test_score_aggregation():
    """Controlled outputs → known NTA hits → assert aggregates + per-category."""
    print("[score aggregation]")
    recs = [
        # NEO: noisy misses gold, clean hits -> Δ +100
        dict(uid="a", table="A", category="NEO", src_noisy="s", src_clean="s2",
             ref_en="r", noise_span=["牛马"], gold_en=["corporate drone"],
             hyp_noisy="a bad literal cattle horse", hyp_clean="life of a corporate drone"),
        # HOM: both hit -> Δ 0
        dict(uid="b", table="A", category="HOM", src_noisy="s", src_clean="s2",
             ref_en="r", noise_span=["x"], gold_en=["the goat"],
             hyp_noisy="you are the GOAT", hyp_clean="you are the GOAT"),
        # NEO with empty gold -> not NTA-scoreable
        dict(uid="c", table="A", category="NEO", src_noisy="s", src_clean="s2",
             ref_en="r", noise_span=["y"], gold_en=[],
             hyp_noisy="whatever", hyp_clean="whatever"),
    ]
    p = tempfile.mktemp(suffix=".jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    r_, cats, nta_idx, s = score_file(p, nta_only=True)
    rows = {row["scope"]: row for row in summarize("m", r_, cats, nta_idx, s, nta_only=True)}
    os.unlink(p)
    allr = rows["all"]
    check(allr["n"] == 3 and allr["n_nta"] == 2, "all: n=3 n_nta=2 (empty gold excluded)")
    check(allr["nta_noisy"] == 50.0, "all NTA noisy = 50 (1 of 2 hit)")
    check(allr["nta_clean"] == 100.0, "all NTA clean = 100")
    check(allr["nta_delta"] == 50.0, "all NTA Δ = +50")
    check(allr["xcomet_delta"] is None, "nta_only -> xcomet None")
    check(rows["NEO"]["nta_delta"] == 100.0, "NEO Δ = +100 (only scoreable NEO)")
    check(rows["HOM"]["nta_delta"] == 0.0, "HOM Δ = 0")


def test_provenance():
    """Manifest structure + the hard rule: never leak a secret's value."""
    print("[provenance]")
    import lib_provenance as P
    fp1 = P.data_fingerprint(FIXTURE, n_records=7)
    fp2 = P.data_fingerprint(FIXTURE, n_records=7)
    check(len(fp1["sha256"]) == 64 and fp1["sha256"] == fp2["sha256"], "data_fingerprint deterministic sha256")
    check(fp1["n_records"] == 7, "fingerprint records n_records")
    m = P.build_manifest("translate", model_key="dummy", data_path=FIXTURE, n_records=7)
    check(m["kind"] == "translate" and "protocol" in m and "env" in m, "manifest has kind/protocol/env")
    check("api_keys_present" in m["env"], "env records API-key presence")
    saved = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-SECRET-do-not-leak-123"
    try:
        blob = json.dumps(P.build_manifest("score", extra={"x": 1}))
        check("sk-SECRET-do-not-leak-123" not in blob, "manifest NEVER leaks a key value")
        check(P.env_info()["api_keys_present"]["OPENAI_API_KEY"] is True, "key presence recorded as bool")
    finally:
        if saved is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = saved


def test_end_to_end():
    """dummy backend → run_score --nta-only → make_tables, on the fixture."""
    print("[end-to-end dummy]")
    tmp = tempfile.mkdtemp()
    out, res = os.path.join(tmp, "outputs"), os.path.join(tmp, "results")
    env = dict(os.environ)

    def run(cmd):
        r = subprocess.run([sys.executable, *cmd], cwd=HERE, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"cmd {cmd} failed:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    run(["run_translate.py", "--data", FIXTURE, "--models", "dummy", "--outdir", out])
    check(os.path.exists(os.path.join(out, "dummy.jsonl")), "run_translate wrote dummy.jsonl")
    check(os.path.exists(os.path.join(out, "dummy.manifest.json")), "run_translate wrote manifest")
    recs = [json.loads(l) for l in open(os.path.join(out, "dummy.jsonl"), encoding="utf-8")]
    check(len(recs) == 7 and "hyp_noisy" in recs[0], "outputs have 7 rows + hyp fields")

    run(["run_score.py", "--nta-only", "--outdir", out, "--resdir", res])
    summ = [json.loads(l) for l in open(os.path.join(res, "summary.jsonl"), encoding="utf-8")]
    scopes = {r["scope"] for r in summ if r["model"] == "dummy"}
    check({"all", "HOM", "PYA", "NEO", "MIX"} <= scopes, "summary has all + 4 category scopes")
    check(os.path.exists(os.path.join(res, "scoring.manifest.json")), "run_score wrote scoring manifest")

    run(["make_tables.py", "--resdir", res])
    with open(os.path.join(res, "table_main.tsv"), encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    check(rows[0][0] == "model" and any(r[0] == "dummy" for r in rows[1:]),
          "table_main.tsv has header + dummy row")
    check(os.path.exists(os.path.join(res, "table_by_category.tsv")), "by-category table written")
    check(os.path.exists(os.path.join(res, "results.md")), "results.md written")


if __name__ == "__main__":
    test_metrics()
    test_load()
    test_score_aggregation()
    test_provenance()
    test_end_to_end()
    print(f"\nALL {_n} CHECKS PASSED")
