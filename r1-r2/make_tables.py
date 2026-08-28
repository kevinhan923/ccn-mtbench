"""Build result tables from results/summary.jsonl.

Outputs:
  results/table_main.tsv          per system (scope=all): noisy/clean/Δ/CI, rel-Δ, NTA
  results/table_by_category.tsv   per system × noise category: XCOMET Δ, NTA Δ
  results/results.md              readable summary

RQ1 reads table_main (system Δ ranking + CI + relative Δ); RQ2 reads
table_by_category (which noise type degrades most).
"""
import csv
import json
import os

from lib_data import CATEGORIES, RESULTS

# specialized MT -> commercial -> open LLM -> frontier: the four tiers RQ1 compares
MODEL_ORDER = ["nllb-3.3b", "google-translate", "qwen3-1.7b", "qwen3-8b",
               "qwen3-32b", "gpt-4o", "claude", "gemini", "dummy"]


def f(v, prec=2):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else ""


def order_key(model):
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else 99, model)


def load_summary(resdir=RESULTS):
    path = os.path.join(resdir, "summary.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return {(r["model"], r["scope"]): r for r in rows}


def models_in(rows):
    return sorted({m for (m, s) in rows if s == "all"}, key=order_key)


def build_main(rows, resdir=RESULTS):
    models = models_in(rows)
    header = ["model", "n", "xcomet_noisy", "xcomet_clean", "xcomet_delta",
              "xcomet_ci", "rel_delta", "kiwi_delta", "nta_noisy", "nta_clean", "nta_delta"]
    lines = [header]
    for m in models:
        r = rows[(m, "all")]
        ci = (f"[{f(r.get('xcomet_delta_lo'))}, {f(r.get('xcomet_delta_hi'))}]"
              if r.get("xcomet_delta_lo") is not None else "")
        lines.append([m, r["n"], f(r.get("xcomet_noisy")), f(r.get("xcomet_clean")),
                      f(r.get("xcomet_delta")), ci, f(r.get("rel_delta"), 3),
                      f(r.get("kiwi_delta")), f(r.get("nta_noisy")), f(r.get("nta_clean")),
                      f(r.get("nta_delta"))])
    out = os.path.join(resdir, "table_main.tsv")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(lines)
    print(f"[write] {out} ({len(models)} systems)")
    return lines


def build_by_category(rows, resdir=RESULTS):
    models = models_in(rows)
    header = ["model"]
    for cat in CATEGORIES:
        header += [f"{cat}_n", f"{cat}_xcomet_delta", f"{cat}_nta_delta"]
    lines = [header]
    for m in models:
        row = [m]
        for cat in CATEGORIES:
            r = rows.get((m, cat))
            if r:
                row += [r["n"], f(r.get("xcomet_delta")), f(r.get("nta_delta"))]
            else:
                row += ["", "", ""]
        lines.append(row)
    out = os.path.join(resdir, "table_by_category.tsv")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(lines)
    print(f"[write] {out}")
    return lines


def build_md(rows, resdir=RESULTS):
    models = models_in(rows)
    out = os.path.join(resdir, "results.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Results: noise-induced degradation (Δ = clean − noisy)\n\n")
        fh.write("## RQ1 — overall (higher Δ = less robust)\n\n")
        fh.write("| system | XCOMET Δ [95% CI] | rel Δ | NTA Δ | Kiwi Δ |\n|---|---|---|---|---|\n")
        for m in models:
            r = rows[(m, "all")]
            ci = (f" [{f(r.get('xcomet_delta_lo'))}, {f(r.get('xcomet_delta_hi'))}]"
                  if r.get("xcomet_delta_lo") is not None else "")
            fh.write(f"| {m} | {f(r.get('xcomet_delta'))}{ci} | {f(r.get('rel_delta'), 3)} "
                     f"| {f(r.get('nta_delta'))} | {f(r.get('kiwi_delta'))} |\n")
        fh.write("\n## RQ2 — by noise category (XCOMET Δ)\n\n")
        fh.write("| system | " + " | ".join(CATEGORIES) + " |\n")
        fh.write("|---" * (1 + len(CATEGORIES)) + "|\n")
        for m in models:
            cells = []
            for cat in CATEGORIES:
                r = rows.get((m, cat))
                cells.append(f(r.get("xcomet_delta")) if r else "")
            fh.write(f"| {m} | " + " | ".join(cells) + " |\n")
        fh.write("\n*Δ in XCOMET/NTA points; positive = the system does worse on noisy input. "
                 "NTA is a hit-rate (0–100). Only NTA populated when scored with --nta-only.*\n")
    print(f"[write] {out}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdir", default=RESULTS)
    args = ap.parse_args()
    rows = load_summary(args.resdir)
    build_main(rows, args.resdir)
    build_by_category(rows, args.resdir)
    build_md(rows, args.resdir)


if __name__ == "__main__":
    main()
