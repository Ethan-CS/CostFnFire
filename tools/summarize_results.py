"""
Summarize all results.csv files under an output/<exp-id>/ directory into a single CSV.
Usage:
  python3 tools/summarize_results.py --base output/<exp-id> [--out summary.csv]
Programmatic API:
  summarize(Path('output/<exp-id>')) -> list of rows written (including header)
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List


HEADER = [
    "graph_type",
    "num_vertices",
    "parameter",
    "seed",
    "outbreak",
    "budget",
    "cost_function",
    "heuristic",
    "num_vertices_saved",
]


def summarize(base_dir: Path, out_path: Path | None = None) -> List[List[str]]:
    rows: List[List[str]] = [HEADER]
    for results_file in base_dir.rglob('results.csv'):
        try:
            with results_file.open('r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                # Try to tolerate minor header differences; map by index if possible
                if header is None:
                    continue
                # If header matches exactly, trust order; else try to re-map
                if [h.strip() for h in header] == HEADER:
                    for r in reader:
                        if r and any(cell.strip() for cell in r):
                            rows.append(r)
                else:
                    # Build index map
                    idx = {name: header.index(name) for name in HEADER if name in header}
                    for r in reader:
                        if not r or not any(cell.strip() for cell in r):
                            continue
                        out_r = [r[idx[name]] if name in idx and idx[name] < len(r) else '' for name in HEADER]
                        rows.append(out_r)
        except Exception as e:
            print(f"Warning: failed to read {results_file}: {e}")

    if out_path is None:
        out_path = base_dir / 'summary.csv'
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    except Exception as e:
        print(f"Warning: failed to write {out_path}: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Summarize simulation results into a single CSV")
    ap.add_argument('--base', required=True, help='Base directory (e.g., output/<exp-id>)')
    ap.add_argument('--out', default=None, help='Output CSV path (default: <base>/summary.csv)')
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out) if args.out else None
    rows = summarize(base, out)
    print(f"Wrote {len(rows)-1} result rows.")


if __name__ == '__main__':
    main()

