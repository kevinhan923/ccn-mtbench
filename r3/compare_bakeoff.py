#!/usr/bin/env python3
"""Print the qwen3-32b normalizer bake-off next to GPT-4o's numbers.

Both arms ran the SAME 32 stratified sentences (4 categories x 4 rows x 2 tables)
through the SAME frozen prompts, so the columns are directly comparable. The
question is not "which restores more" — 32 rows cannot answer that — it is
**"can qwen3-32b follow the output contract at all"**, because if it cannot, the
A2 - A1 headline stops being the conservative estimate RESEARCH_PLAN v3.7 claimed.

Run from r3/, after the bake-off normalization. Pure local string work.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "r1-r2")))

from lib_data import load_contrastive

NORM = os.path.normpath(os.path.join(HERE, "..", "r3_exp_result", "normalized"))
SAMPLE = os.path.join(NORM, "dryrun", "dryrun_sample32.tsv")

# gpt-4o's a2 comes from the decline variant, which IS the frozen contract
ARMS = {
    "gpt-4o": {"a1": "dryrun/norm_a1", "a2": "dryrun_decline/norm_a2", "a3": "dryrun/norm_a3"},
    "qwen3-32b": {t: f"bakeoff_qwen32b/norm_{t}" for t in ("a1", "a2", "a3")},
}


def read(stem):
    p = os.path.join(NORM, f"{stem}_dryrun_sample32.log.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    recs = {r["uid"]: r for r in load_contrastive(SAMPLE)}
    print(f"bake-off on {len(recs)} stratified sentences "
          "(HOM/PYA/NEO/MIX x 4 x 2 tables), frozen prompts\n")
    print(f"{'arm':12s}{'a1':>8s}{'a2':>8s}{'a3':>8s}{'parse fail':>13s}{'echo ok':>10s}")

    summary = {}
    for arm, srcs in ARMS.items():
        cells, fails, calls, echo_ok, echo_n = [], 0, 0, 0, 0
        missing = False
        for tier, stem in srcs.items():
            log = read(stem)
            if log is None:
                missing = True
                cells.append("  --")
                continue
            called = [e for e in log if e["llm_raw"] is not None]
            ex = sum(e["src_norm"] == recs[e["uid"]]["src_clean"] for e in log)
            cells.append(f"{ex:2d}/32")
            fails += sum(not e["parse_ok"] for e in log)
            calls += len(called)
            if tier in ("a2", "a3"):          # only these two have an echo contract
                echo_n += len(called)
                echo_ok += sum(e["parse_ok"] for e in called)
        if missing and arm == "qwen3-32b":
            print(f"  {arm:10s} not run yet — see RUN_R3_QWEN.md step 2")
            continue
        pct = f"{100*echo_ok/echo_n:.0f}%" if echo_n else "  -"
        print(f"{arm:12s}" + "".join(f"{c:>8s}" for c in cells)
              + f"{f'{fails}/{calls}':>13s}{pct:>10s}")
        summary[arm] = (fails, calls, cells)

    if "qwen3-32b" not in summary:
        return
    g_f, g_c, g_cells = summary["gpt-4o"]
    q_f, q_c, q_cells = summary["qwen3-32b"]
    print("\n" + "-" * 60)
    g_rate = g_f / g_c if g_c else 0
    q_rate = q_f / q_c if q_c else 0
    a1_g = int(g_cells[0].split("/")[0])
    a1_q = int(q_cells[0].split("/")[0])
    print(f"parse-failure rate: gpt-4o {g_rate:.1%}  vs  qwen3-32b {q_rate:.1%}")
    print(f"a1 (the baseline that carries the conservatism claim): {a1_g}/32 vs {a1_q}/32")
    print()
    if q_rate > g_rate + 0.05 or a1_q < a1_g - 3:
        print("VERDICT: qwen3-32b is materially weaker on the contract or on A1.")
        print("  A2 - A1 on this arm is NOT a conservative estimate. The paper must")
        print("  say so plainly. Do not tune the prompt — it is frozen (red line 3).")
    else:
        print("VERDICT: qwen3-32b matches GPT-4o closely enough on this sample.")
        print("  The conservatism argument survives; state that it was checked, and")
        print("  note that n=32 bounds how strongly it can be claimed.")


if __name__ == "__main__":
    main()
