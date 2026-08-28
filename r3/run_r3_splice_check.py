"""Pre-validate the R3 splice premise on the benchmark (RESEARCH_PLAN §R3).

For every contrastive row, splicing noise_span -> noise_std into src_noisy
must reproduce src_clean exactly. Rows that fail make A4 != clean side and
muddy the A4-A3 "restoration error" reading, so we must know them up front.
Pure local string work — no models.

Usage (run from r3/):
  python run_r3_splice_check.py                     # both tables, report to r3/reports/
  python run_r3_splice_check.py --data ../r1-r2/data/contrastive_A.tsv
"""
import argparse
import csv
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
_R1R2 = os.path.normpath(os.path.join(HERE, "..", "r1-r2"))
if _R1R2 not in sys.path:
    sys.path.insert(0, _R1R2)

from lib_data import DATA, load_contrastive
from lib_r3 import gold_splice

DEFAULT_TABLES = ["contrastive_A.tsv", "contrastive_B.tsv"]
REPORTS = os.path.join(HERE, "reports")


def check_table(path: str) -> list[dict]:
    rows = []
    for r in load_contrastive(path):
        status, spliced, detail = "ok", "", ""
        try:
            spliced = gold_splice(r)
        except ValueError as e:
            status, detail = "splice_error", str(e)
        else:
            if spliced != r["src_clean"]:
                if (unicodedata.normalize("NFKC", spliced)
                        == unicodedata.normalize("NFKC", r["src_clean"])):
                    status, detail = "nfkc_only", "differs only under NFKC"
                else:
                    status, detail = "mismatch", f"spliced={spliced!r}"
        rows.append({"uid": r["uid"], "category": r["category"], "status": status,
                     "src_noisy": r["src_noisy"], "src_clean": r["src_clean"],
                     "spliced": spliced, "detail": detail})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", help="TSV path(s); default both tables")
    ap.add_argument("--report", default=os.path.join(REPORTS, "r3_splice_check_20260729.tsv"))
    args = ap.parse_args()
    paths = args.data or [os.path.join(DATA, t) for t in DEFAULT_TABLES]
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    all_rows = []
    for p in paths:
        rows = check_table(p)
        n_bad = sum(r["status"] != "ok" for r in rows)
        print(f"{p}: n={len(rows)}  ok={len(rows) - n_bad}  problems={n_bad}")
        for r in rows:
            if r["status"] != "ok":
                print(f"  [{r['status']}] {r['uid']}: {r['detail']}")
        all_rows.extend(rows)

    with open(args.report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()),
                           delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)
    n_bad = sum(r["status"] != "ok" for r in all_rows)
    print(f"[write] {args.report} ({len(all_rows)} rows, {n_bad} problems)")
    if n_bad == 0:
        print("SPLICE PRECONDITION HOLDS: gold splice reproduces src_clean on every row.")


if __name__ == "__main__":
    main()
