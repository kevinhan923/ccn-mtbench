"""Local self-test for the R3 mark-and-translate plumbing (no models needed).

Covers: span/std pairing, span location, right-to-left splice, gold splice
(A4 identity), tau abstention, LLM span-output parsing (A1 / A2 normalizer),
detector-JSONL interface validation, the run_r3_normalize dummy end-to-end,
and make_tables_r3 ladder/comparison arithmetic on controlled numbers.
Run:  python selftest_r3.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import lib_r3 as R

HERE = os.path.dirname(os.path.abspath(__file__))

_n = 0


def check(cond, msg):
    global _n
    _n += 1
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ok: {msg}")


def raises(fn, msg):
    try:
        fn()
    except (ValueError, KeyError):
        check(True, msg)
    else:
        check(False, msg)


def test_pair_spans_stds():
    print("[pair_spans_stds]")
    check(R.pair_spans_stds(["牛马"], ["打工人"]) == [("牛马", "打工人")], "1:1 pair")
    check(R.pair_spans_stds(["a", "b"], ["x", "y"]) == [("a", "x"), ("b", "y")],
          "2:2 pair keeps order")
    raises(lambda: R.pair_spans_stds(["a", "b"], ["x"]), "count mismatch raises")
    raises(lambda: R.pair_spans_stds([], []), "empty spans raises")


def test_locate_spans():
    print("[locate_spans]")
    check(R.locate_spans("广0王，我也是", ["广0王"]) == [(0, 3, "广0王")], "simple find")
    # two identical span texts claim the 1st and 2nd occurrences
    got = R.locate_spans("哈yyds哈yyds", ["yyds", "yyds"])
    check(got == [(1, 5, "yyds"), (6, 10, "yyds")], "duplicate span claims both occurrences")
    # spans given out of order still return sorted by start
    got = R.locate_spans("A和B", ["B", "A"])
    check(got == [(0, 1, "A"), (2, 3, "B")], "result sorted by start")
    raises(lambda: R.locate_spans("abc", ["x"]), "span not found raises")
    raises(lambda: R.locate_spans("abc", ["ab", "bc"]), "overlapping spans raise")


def test_splice():
    print("[splice]")
    check(R.splice("3Q 新年快乐", [(0, 2, "谢谢")]) == "谢谢 新年快乐", "single replace")
    # right-to-left: earlier replacement must not shift later offsets
    s = "x牛马y绝绝子z"
    out = R.splice(s, [(1, 3, "打工人"), (4, 7, "绝了")])
    check(out == "x打工人y绝了z", "multi replace with length change")
    check(R.splice("ab", [(0, 1, "A"), (1, 2, "B")]) == "AB", "adjacent spans ok")
    check(R.splice("abc", []) == "abc", "no replacements = identity")
    raises(lambda: R.splice("abc", [(0, 2, "x"), (1, 3, "y")]), "overlap raises")
    raises(lambda: R.splice("abc", [(1, 9, "x")]), "out of range raises")
    raises(lambda: R.splice("abc", [(2, 1, "x")]), "start>end raises")


def test_gold_splice():
    print("[gold_splice]")
    rec = {"src_noisy": "周末加班，真的是牛马命", "src_clean": "周末加班，真的是打工人的命",
           "noise_span": ["牛马"], "noise_std": "打工人的"}
    check(R.gold_splice(rec) == rec["src_clean"], "gold splice reproduces src_clean")
    rec2 = {"src_noisy": "3Q，yyds", "src_clean": "谢谢，永远的神",
            "noise_span": ["3Q", "yyds"], "noise_std": "谢谢‖永远的神"}
    check(R.gold_splice(rec2) == rec2["src_clean"], "multi-span gold splice")
    # span listed once but occurring twice: gold splice replaces ALL occurrences
    # (annotator convention, e.g. B-PYA-053 'bt...bt' -> '变态...变态')
    rec3 = {"src_noisy": "bt，不够bt", "src_clean": "变态，不够变态",
            "noise_span": ["bt"], "noise_std": "变态"}
    check(R.gold_splice(rec3) == rec3["src_clean"], "repeated occurrence replaced everywhere")
    # a std that contains another span's text must not be re-replaced
    rec4 = {"src_noisy": "a和b", "src_clean": "xb和y",
            "noise_span": ["a", "b"], "noise_std": "xb‖y"}
    check(R.gold_splice(rec4) == rec4["src_clean"], "std introducing a span text is safe")


def test_apply_tau():
    print("[apply_tau]")
    spans = [{"text": "a", "confidence": 0.9}, {"text": "b", "confidence": 0.4}]
    check(R.apply_tau(spans, None) == spans, "tau None keeps all")
    check(R.apply_tau(spans, 0.5) == [spans[0]], "tau filters below threshold")
    check(R.apply_tau(spans, 0.9) == [spans[0]], "tau keeps >= threshold")
    raises(lambda: R.apply_tau([{"text": "a"}], 0.5), "missing confidence raises when tau set")


def test_parse_llm_spans():
    print("[parse_llm_spans]")
    raw = '{"spans": [{"text": "蚌埠住了", "type": "HOM", "std": "绷不住了"}]}'
    got = R.parse_llm_spans(raw)
    check(got == [{"text": "蚌埠住了", "type": "HOM", "std": "绷不住了"}], "plain JSON")
    fenced = "```json\n" + raw + "\n```"
    check(R.parse_llm_spans(fenced) == got, "strips code fence")
    check(R.parse_llm_spans('{"spans": []}') == [], "empty spans = no noise")
    raises(lambda: R.parse_llm_spans("I think there is no noise."), "non-JSON raises")
    raises(lambda: R.parse_llm_spans('{"spans": [{"text": "x", "type": "BAD", "std": "y"}]}'),
           "unknown type raises")
    raises(lambda: R.parse_llm_spans('{"spans": [{"type": "HOM"}]}'),
           "missing text/std raises")
    # LLM-produced std goes into a TSV: control characters must be rejected
    raises(lambda: R.parse_llm_spans('{"spans": [{"text": "x", "type": "HOM", "std": "a\\tb"}]}'),
           "std with tab raises (would corrupt TSV)")
    raises(lambda: R.parse_llm_spans('{"spans": [{"text": "x", "type": "HOM", "std": "a\\nb"}]}'),
           "std with newline raises")
    raises(lambda: R.parse_llm_spans('{"spans": [{"text": "x", "type": "HOM", "std": 123}]}'),
           "non-string std raises")


def test_gold_span_ranges():
    print("[gold_span_ranges]")
    rec = {"src_noisy": "bt,不够bt", "noise_span": ["bt"], "noise_std": "变态"}
    got = R.gold_span_ranges(rec)
    check(got == [(0, 2, "bt", "变态"), (5, 7, "bt", "变态")],
          "all occurrences claimed, std attached, sorted by start")
    rec2 = {"src_noisy": "3Q,yyds", "noise_span": ["3Q", "yyds"], "noise_std": "谢谢‖永远的神"}
    check(R.gold_span_ranges(rec2) == [(0, 2, "3Q", "谢谢"), (3, 7, "yyds", "永远的神")],
          "multi-span pairing preserved")


def test_span_detection_stats():
    print("[span_detection_stats]")
    gold = [(0, 2, "bt", "变态"), (5, 7, "bt", "变态")]
    pred = [{"start": 0, "end": 2, "text": "bt", "type": "PYA", "confidence": 0.9},
            {"start": 3, "end": 4, "text": ",", "type": "MIX", "confidence": 0.5}]
    s = R.span_detection_stats(gold, pred)
    check(s["n_gold"] == 2 and s["n_pred"] == 2, "counts")
    check(s["strict_tp"] == 1, "one exact-boundary match")
    check(s["matched"] == [(0, 0)], "matched pair indices (gold_idx, pred_idx)")
    check(s["gold_chars"] == 4 and s["pred_chars"] == 3 and s["overlap_chars"] == 2,
          "char-level sets: gold 4, pred 3, overlap 2")
    s2 = R.span_detection_stats(gold, [])
    check(s2["strict_tp"] == 0 and s2["overlap_chars"] == 0, "no predictions -> zeros")


def test_std_match():
    print("[std_match]")
    check(R.std_match("永远的神", "永远的神") == (True, True), "exact match")
    check(R.std_match("不好意", "不好意思") == (False, True), "fuzzy-only match (substring)")
    check(R.std_match("完全不同", "永远的神") == (False, False), "no match")
    check(R.std_match(" 永远的神 ", "永远的神")[0], "exact is whitespace/case-normalized")


def _write_detector(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_validate_detector_jsonl():
    print("[validate_detector_jsonl]")
    records = [
        {"uid": "A-HOM-001", "src_noisy": "广0王，我也是"},
        {"uid": "A-PYA-001", "src_noisy": "yyds真的"},
    ]
    good = [
        {"uid": "A-HOM-001", "src_noisy": "广0王，我也是",
         "spans": [{"start": 0, "end": 3, "text": "广0王", "type": "HOM",
                    "confidence": 0.93, "candidates": ["广陵王"]}],
         "model_version": "fuse4-gen2-test"},
        {"uid": "A-PYA-001", "src_noisy": "yyds真的",
         "spans": [{"start": 0, "end": 4, "text": "yyds", "type": "PYA",
                    "confidence": 0.88}],
         "model_version": "fuse4-gen2-test"},
    ]
    p = tempfile.mktemp(suffix=".jsonl")
    _write_detector(p, good)
    rep = R.validate_detector_jsonl(p, records)
    check(rep["ok"] and rep["n"] == 2 and rep["n_spans"] == 2, "good file passes")
    os.unlink(p)

    bad_cases = [
        (good[:1], "missing uid coverage"),                      # A-PYA-001 absent
        ([dict(good[0], spans=[dict(good[0]["spans"][0], text="广陵王")]), good[1]],
         "span text != src_noisy[start:end]"),
        ([dict(good[0], spans=[dict(good[0]["spans"][0], confidence=1.5)]), good[1]],
         "confidence out of [0,1]"),
        ([dict(good[0], spans=[dict(good[0]["spans"][0], type="SLANG")]), good[1]],
         "type outside HOM/PYA/NEO/MIX"),
        ([dict(good[0], src_noisy="改了"), good[1]], "src_noisy mismatch vs benchmark"),
        ([dict(good[0], spans=[good[0]["spans"][0],
                               {"start": 1, "end": 4, "text": "0王，", "type": "HOM",
                                "confidence": 0.5}]), good[1]],
         "overlapping spans within a sentence"),
    ]
    for rows, why in bad_cases:
        p = tempfile.mktemp(suffix=".jsonl")
        _write_detector(p, rows)
        rep = R.validate_detector_jsonl(p, records)
        check(not rep["ok"] and rep["errors"], f"flags {why}")
        os.unlink(p)

    # One detector file covers all of E; the pipeline runs one table at a time.
    both_tables = good + [
        {"uid": "B-NEO-001", "src_noisy": "中午去干饭",
         "spans": [{"start": 3, "end": 5, "text": "干饭", "type": "NEO",
                    "confidence": 0.71}],
         "model_version": "fuse4-gen2-test"},
    ]
    p = tempfile.mktemp(suffix=".jsonl")
    _write_detector(p, both_tables)
    rep = R.validate_detector_jsonl(p, records)
    check(not rep["ok"] and any("unknown uid" in e for e in rep["errors"]),
          "strict mode flags uids outside the given table (acceptance check)")
    rep = R.validate_detector_jsonl(p, records, allow_extra_uids=True)
    check(rep["ok"] and rep["n"] == 2 and rep["n_spans"] == 2,
          "allow_extra_uids skips the other table's lines, spans not counted")
    os.unlink(p)

    # allow_extra_uids must NOT weaken the coverage requirement
    p = tempfile.mktemp(suffix=".jsonl")
    _write_detector(p, [good[0], both_tables[2]])
    rep = R.validate_detector_jsonl(p, records, allow_extra_uids=True)
    check(not rep["ok"] and any("missing" in e for e in rep["errors"]),
          "allow_extra_uids still requires full coverage of the given table")
    os.unlink(p)


MINI_TSV = (
    "uid\ttable\tcategory\tsrc_noisy\tsrc_clean\tref_en\tnoise_span\tnoise_gold_en\tnoise_std\tsource_meta\n"
    "T-HOM-001\tA\tHOM\t童鞋们好\t同学们好\tHello classmates\t童鞋\tclassmates\t同学\ttest\n"
    "T-PYA-001\tA\tPYA\tbhys打扰了\t不好意思打扰了\tSorry to bother\tbhys\tsorry\t不好意思\ttest\n"
    "T-NEO-001\tA\tNEO\t中午去干饭\t中午去吃饭\tGoing to eat at noon\t干饭\teat\t吃饭\ttest\n"
    "T-MIX-001\tA\tMIX\thold住场面再hold住\t稳住场面再稳住\tHold the scene\thold住\thold\t稳住\ttest\n"
)


def _run(cmd, env=None):
    r = subprocess.run([sys.executable, *cmd], cwd=HERE, capture_output=True,
                       text=True, env=env)
    if r.returncode != 0:
        raise AssertionError(f"cmd {cmd} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def _read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_normalize_dummy():
    """run_r3_normalize.py: a4 identity, a2 detector+tau, a1 dummy no-noise."""
    print("[run_r3_normalize dummy]")
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "mini.tsv")
    open(data, "w", encoding="utf-8").write(MINI_TSV)

    # a4: no LLM; output src_noisy column must equal input src_clean everywhere
    out = os.path.join(tmp, "o_a4")
    _run(["run_r3_normalize.py", "--tier", "a4", "--data", data, "--outdir", out])
    rows = _read_tsv(os.path.join(out, "norm_a4_mini.tsv"))
    check(len(rows) == 4, "a4: 4 rows out")
    check(all(r["src_noisy"] == r["src_clean"] for r in rows),
          "a4: normalized == src_clean (identity)")

    # a2: detector spans + dummy normalizer (std = '☆'+text); tau filters one span
    det = os.path.join(tmp, "det.jsonl")
    recs = _read_tsv(data)
    with open(det, "w", encoding="utf-8") as f:
        for r in recs:
            sp = r["noise_span"]
            start = r["src_noisy"].find(sp)
            conf = 0.3 if r["uid"] == "T-NEO-001" else 0.9
            f.write(json.dumps({
                "uid": r["uid"], "src_noisy": r["src_noisy"],
                "spans": [{"start": start, "end": start + len(sp), "text": sp,
                           "type": r["category"], "confidence": conf}],
                "model_version": "dummy-det"}, ensure_ascii=False) + "\n")
    out = os.path.join(tmp, "o_a2")
    _run(["run_r3_normalize.py", "--tier", "a2", "--data", data, "--detector", det,
          "--tau", "0.5", "--backend", "dummy", "--outdir", out])
    rows = {r["uid"]: r for r in _read_tsv(os.path.join(out, "norm_a2_mini.tsv"))}
    check(rows["T-HOM-001"]["src_noisy"] == "☆童鞋们好", "a2: span replaced by dummy std")
    check(rows["T-NEO-001"]["src_noisy"] == "中午去干饭",
          "a2: tau-filtered span left unnormalized")
    check(rows["T-MIX-001"]["src_noisy"].count("☆") == 1,
          "a2: detector offsets replace only the marked occurrence")
    log = [json.loads(l) for l in open(os.path.join(out, "norm_a2_mini.log.jsonl"),
                                       encoding="utf-8")]
    check(len(log) == 4 and all(e["parse_ok"] for e in log), "a2: log has 4 parse-ok rows")
    check(os.path.exists(os.path.join(out, "norm_a2_mini.manifest.json")),
          "a2: manifest written")

    # a2 + --limit: dry-run subset must work against a FULL-coverage detector file
    out = os.path.join(tmp, "o_a3lim")
    _run(["run_r3_normalize.py", "--tier", "a2", "--data", data, "--detector", det,
          "--backend", "dummy", "--outdir", out, "--limit", "2"])
    rows = _read_tsv(os.path.join(out, "norm_a2_mini.tsv"))
    check(len(rows) == 2, "a2 --limit 2 works with full detector file")

    # a3: gold spans + dummy normalizer; repeated occurrence replaced everywhere
    out = os.path.join(tmp, "o_a3")
    _run(["run_r3_normalize.py", "--tier", "a3", "--data", data,
          "--backend", "dummy", "--outdir", out])
    rows = {r["uid"]: r for r in _read_tsv(os.path.join(out, "norm_a3_mini.tsv"))}
    check(rows["T-HOM-001"]["src_noisy"] == "☆童鞋们好", "a3: gold span normalized")
    check(rows["T-MIX-001"]["src_noisy"] == "☆hold住场面再☆hold住",
          "a3: gold span replaces ALL occurrences")

    # a1: dummy self-detection finds nothing -> identity, parse_ok
    out = os.path.join(tmp, "o_a1")
    _run(["run_r3_normalize.py", "--tier", "a1", "--data", data,
          "--backend", "dummy", "--outdir", out])
    rows = _read_tsv(os.path.join(out, "norm_a1_mini.tsv"))
    check(all(r["src_noisy"] == m["src_noisy"] for r, m in zip(rows, recs)),
          "a1: dummy finds no noise -> identity")


def test_detector_eval():
    """Controlled detector + norm-log -> known detection/std numbers per scope."""
    print("[run_r3_detector_eval]")
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "mini.tsv")
    open(data, "w", encoding="utf-8").write(
        "uid\ttable\tcategory\tsrc_noisy\tsrc_clean\tref_en\tnoise_span\tnoise_gold_en\tnoise_std\tsource_meta\n"
        "T-HOM-001\tA\tHOM\t童鞋们好\t同学们好\tHello classmates\t童鞋\tclassmates\t同学\ttest\n"
        "T-PYA-001\tA\tPYA\tbhys打扰了\t不好意思打扰了\tSorry to bother\tbhys\tsorry\t不好意思\ttest\n")
    det = os.path.join(tmp, "det.jsonl")
    _write_detector(det, [
        # exact boundary + right type + candidate hits gold std
        {"uid": "T-HOM-001", "src_noisy": "童鞋们好",
         "spans": [{"start": 0, "end": 2, "text": "童鞋", "type": "HOM",
                    "confidence": 0.9, "candidates": ["同学", "童靴"]}],
         "model_version": "t"},
        # wrong boundary (bhy vs bhys): no strict match, 3/4 char overlap
        {"uid": "T-PYA-001", "src_noisy": "bhys打扰了",
         "spans": [{"start": 0, "end": 3, "text": "bhy", "type": "PYA",
                    "confidence": 0.8}],
         "model_version": "t"}])
    log = os.path.join(tmp, "norm_a3.log.jsonl")
    with open(log, "w", encoding="utf-8") as f:
        f.write(json.dumps({"uid": "T-HOM-001", "tier": "a3", "parse_ok": True,
                            "spans": [{"text": "童鞋", "type": "HOM", "std": "同学"}]},
                           ensure_ascii=False) + "\n")
        f.write(json.dumps({"uid": "T-PYA-001", "tier": "a3", "parse_ok": True,
                            "spans": [{"text": "bhys", "type": "PYA", "std": "不好意"}]},
                           ensure_ascii=False) + "\n")
    out = os.path.join(tmp, "eval")
    _run(["run_r3_detector_eval.py", "--data", data, "--detector", det,
          "--norm-log", f"a3={log}", "--outdir", out])

    d = {r["scope"]: r for r in _read_tsv(os.path.join(out, "eval_detection.tsv"))}
    check(d["all"]["strict_p"] == "50.00" and d["all"]["strict_r"] == "50.00",
          "strict span P/R = 50 (1 of 2 exact)")
    check(d["all"]["char_p"] == "100.00" and d["all"]["char_r"] == "83.33",
          "char-level P=100, R=83.33 (5/6 gold chars covered)")
    check(d["all"]["type_acc"] == "100.00", "type accuracy on matched spans = 100")
    check(d["HOM"]["strict_r"] == "100.00" and d["PYA"]["strict_r"] == "0.00",
          "per-category recall split (HOM found, PYA boundary-missed)")
    check(d["HOM"]["cand_hit1"] == "100.00", "HOM candidate hit@1 = 100")

    s = {r["scope"]: r for r in _read_tsv(os.path.join(out, "eval_std_a3.tsv"))}
    check(s["all"]["exact_rate"] == "50.00" and s["all"]["fuzzy_rate"] == "100.00",
          "std restoration: exact 50, fuzzy 100")
    check(os.path.exists(os.path.join(out, "detector_eval.md")), "md summary written")


def _write_segments(path, rows):
    cols = ["uid", "category", "xcomet_noisy", "xcomet_clean", "xcomet_d",
            "kiwi_noisy", "kiwi_clean", "kiwi_d", "nta_noisy", "nta_clean", "nta_d",
            "src_noisy", "hyp_noisy", "src_clean", "hyp_clean", "ref_en"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def test_make_tables_r3():
    """Controlled scores -> known ladder / recovery / comparison numbers."""
    print("[make_tables_r3]")
    tmp = tempfile.mkdtemp()
    # A0: noisy 60, clean 90 -> Δ_A0 = 30 (uids u1..u4, two categories)
    a0 = [dict(uid=f"u{i}", category=("HOM" if i < 3 else "NEO"),
               xcomet_noisy="60.00", xcomet_clean="90.00", xcomet_d="30.00",
               nta_noisy="0.00", nta_clean="100.00", nta_d="100.00")
          for i in range(1, 5)]
    # tiers score their normalized side as 'noisy': a1 70, a2 80, a3 85
    tiers = {}
    for name, score in [("a1", 70), ("a2", 80), ("a3", 85)]:
        p = os.path.join(tmp, f"seg_{name}.tsv")
        _write_segments(p, [dict(r, xcomet_noisy=f"{score:.2f}") for r in a0])
        tiers[name] = p
    a0p = os.path.join(tmp, "seg_a0.tsv")
    _write_segments(a0p, a0)
    res = os.path.join(tmp, "res")
    _run(["make_tables_r3.py", "--a0-segments", a0p, "--resdir", res,
          "--tier", f"a1={tiers['a1']}", "--tier", f"a2={tiers['a2']}",
          "--tier", f"a3={tiers['a3']}"])

    ladder = {r["tier"]: r for r in _read_tsv(os.path.join(res, "table_ladder.tsv"))}
    check([ladder[t]["xcomet"] for t in ("a0", "a1", "a2", "a3", "a4")]
          == ["60.00", "70.00", "80.00", "85.00", "90.00"], "ladder scores in order")
    check(ladder["a2"]["recovery"] == "20.00", "recovery(a2) = score - A0 = 20")
    check(ladder["a4"]["delta_vs_clean"] == "0.00", "A4 sits at the clean anchor")

    comps = {r["pair"]: r for r in _read_tsv(os.path.join(res, "comparisons.tsv"))}
    check(comps["a2-a1"]["diff"] == "10.00", "headline a2-a1 diff = 10")
    check(float(comps["a2-a1"]["ci_lo"]) <= 10 <= float(comps["a2-a1"]["ci_hi"]),
          "a2-a1 CI brackets the diff")
    check(comps["a4-a3"]["diff"] == "5.00", "a4-a3 diff = 5")
    check(os.path.exists(os.path.join(res, "table_ladder_by_category.tsv")),
          "per-category ladder written")
    check(os.path.exists(os.path.join(res, "results_r3.md")), "results_r3.md written")

    # anchors are frozen: --tier a0/a4 must be rejected, not silently overwrite
    r = subprocess.run([sys.executable, "make_tables_r3.py", "--a0-segments", a0p,
                        "--resdir", os.path.join(tmp, "res2"),
                        "--tier", f"a4={tiers['a2']}"],
                       cwd=HERE, capture_output=True, text=True)
    check(r.returncode != 0, "make_tables_r3 rejects --tier a4 (anchor override)")


if __name__ == "__main__":
    test_pair_spans_stds()
    test_locate_spans()
    test_splice()
    test_gold_splice()
    test_apply_tau()
    test_parse_llm_spans()
    test_validate_detector_jsonl()
    test_gold_span_ranges()
    test_span_detection_stats()
    test_std_match()
    test_normalize_dummy()
    test_detector_eval()
    test_make_tables_r3()
    print(f"\nALL {_n} CHECKS PASSED")
