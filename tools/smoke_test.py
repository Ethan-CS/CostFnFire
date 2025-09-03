#!/usr/bin/env python3
"""
Run small, fast simulations across several graph types to validate the pipeline locally.
This keeps trials and sizes small so it finishes quickly.

Usage:
  python3 tools/smoke_test.py
Optional args:
  --size 60 --trials 2 --jobs 1 --output-dir output --exp-id SMOKE-<timestamp>
"""
from __future__ import annotations
import argparse
import datetime
from pathlib import Path
import sys

# Ensure project root is on sys.path when invoked from tools/
PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from costs_and_heuristics import CFn, Heuristic, HeuristicChoices
from simulation import run_experiments, write_degrees_to_csv, write_results_to_file


def parse_args():
    ap = argparse.ArgumentParser(description="Run quick smoke tests for simulation")
    ap.add_argument("--size", type=int, default=60)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default="output")
    ap.add_argument("--exp-id", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    # Keep this light: couple of heuristics, couple of costs
    heuristics = [
        Heuristic(HeuristicChoices.RANDOM, None),
        Heuristic(HeuristicChoices.DEGREE, None),
    ]
    cost_functions = [CFn.UNIFORM, CFn.UNIFORMLY_RANDOM]

    graphs = {
        "erdos-renyi": [0.05],
        "barabasi-albert": [2],
        "random geometric": [0.1],
        "random n-regular": [3],
        "watts-strogatz": [4],
        "powerlaw-cluster": [2],
        "scale-free": [-1],
        "connected-caveman": [5],
        # "random-lobster": [-1],
    }

    exp_id = args.exp_id or f"SMOKE-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    print(f"Smoke test: size={args.size} trials={args.trials} jobs={args.jobs} exp={exp_id}")

    for budget in [1, 2]:
        for cost in cost_functions:
            base = Path(args.output_dir) / exp_id / str(cost)
            base.mkdir(parents=True, exist_ok=True)
            for graph, params in graphs.items():
                res_accum = {}
                for p in params:
                    trials, actual_size, degrees, strategies, cost_mappings = run_experiments(
                        size=args.size, p=p, num_trials=args.trials, graph_type=graph, heuristics=heuristics,
                        cost_type=cost, budget=budget, outbreak='rand', progress=False, workers=args.jobs,
                    )
                    res_accum.update(trials)
                    write_degrees_to_csv(degrees, str(base / graph.replace(' ', '_')))
                write_results_to_file(graph, str(base), res_accum)
                print(f"OK: graph={graph} cost={cost} budget={budget} -> {len(res_accum)} rows")

    print("Smoke test finished. Check output/ and logs.")


if __name__ == "__main__":
    main()
