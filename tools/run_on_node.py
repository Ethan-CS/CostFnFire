#!/usr/bin/env python3
"""
Helper launcher for a single machine (e.g., one fatanode).
- Runs one or more graphs sequentially.
- For each graph, uses tools/launch_batch.py to fan out over parameter values with per-process logs.
- Picks sensible concurrency defaults based on available CPUs.

Examples:
  # Run ER on this node, sweeping default params, ~64 workers total as 8 procs x 8 jobs each (on a 64-thread node)
  python3 tools/run_on_node.py \
    --graphs "erdos-renyi" \
    --size 200 --trials 200 --budgets 1,2,3,4,5 \
    --costs "Uniform,Uniformly Random,Hesitancy-Binary" \
    --heuristics "Random,Degree,Threat,Cost" \
    --jobs-per-process 8 --max-procs 8 --progress --resume

  # Split multiple graphs on one node, sequentially
  python3 tools/run_on_node.py --graphs "watts-strogatz,powerlaw-cluster,connected-caveman" --trials 100 --resume

Notes:
- Output goes under output/<exp-id>/<cost-fn>/<graph>/ by simulation.py
- Logs go under logs/<exp-id>/<graph> by launch_batch.py
- Processes are started in their own sessions so they keep running after SSH disconnect
"""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent


def _default_concurrency():
    cpus = os.cpu_count() or 8
    # Aim for jobs_per_process * max_procs ~= cpus
    jobs_per_process = max(1, cpus // 8)  # 8 workers per process on a 64-thread node
    max_procs = max(1, cpus // jobs_per_process)
    return jobs_per_process, max_procs


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Run one or more graph sweeps on a single node using launch_batch.py")
    ap.add_argument("--graphs", type=str, required=True,
                    help="Comma-separated list of graphs (e.g., 'erdos-renyi,barabasi-albert')")
    ap.add_argument("--params", type=str, default=None,
                    help="Optional comma-separated parameter list to use for all graphs; default uses per-graph defaults")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--budgets", type=str, default="1,2,3,4,5")
    ap.add_argument("--costs", type=str, default=None,
                    help="Optional comma-separated cost function names (match simulation.py names)")
    ap.add_argument("--heuristics", type=str, default=None,
                    help="Optional comma-separated heuristics (match simulation.py names)")
    ap.add_argument("--outbreak", type=str, default="rand")
    ap.add_argument("--output-dir", type=str, default="output")
    ap.add_argument("--exp-id", type=str, default=None,
                    help="Experiment id for output and logs; default is timestamp")
    def_jp, def_mp = _default_concurrency()
    ap.add_argument("--jobs-per-process", type=int, default=def_jp,
                    help="--jobs passed to each simulation.py process (default scales with CPUs)")
    ap.add_argument("--max-procs", type=int, default=def_mp,
                    help="Max concurrent simulation.py processes per graph (default scales with CPUs)")
    ap.add_argument("--timeout", type=int, default=0,
                    help="Seconds to kill a simulation process if exceeded (0 = no timeout)")
    ap.add_argument("--save-details", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Skip launching a param if its results appear to already exist")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    graphs = [g.strip() for g in args.graphs.split(',') if g.strip()]
    exp_id = args.exp_id or datetime.now().strftime('%Y%m%d%H%M%S')

    print(f"Node launcher starting for graphs={graphs} exp-id={exp_id}")
    print(f"Concurrency: jobs-per-process={args.jobs_per_process} max-procs={args.max_procs}")

    for g in graphs:
        cmd = [sys.executable, str(THIS_DIR / 'launch_batch.py'),
               '--graph', g,
               '--size', str(args.size),
               '--trials', str(args.trials),
               '--budgets', args.budgets,
               '--outbreak', args.outbreak,
               '--jobs-per-process', str(args.jobs_per_process),
               '--max-procs', str(args.max_procs),
               '--output-dir', args.output_dir,
               '--exp-id', exp_id,
               ]
        if args.params:
            cmd += ['--params', args.params]
        if args.costs:
            cmd += ['--costs', args.costs]
        if args.heuristics:
            cmd += ['--heuristics', args.heuristics]
        if args.save_details:
            cmd += ['--save-details']
        if args.progress:
            cmd += ['--progress']
        if args.resume:
            cmd += ['--resume']
        if args.timeout and args.timeout > 0:
            cmd += ['--timeout', str(args.timeout)]
        if args.dry_run:
            cmd += ['--dry-run']

        print("\n=== Launching graph:", g)
        print("CMD:", shlex.join(cmd))
        if args.dry_run:
            continue
        # Run graphs sequentially to avoid memory spikes; launch_batch does its own concurrency
        proc = subprocess.Popen(cmd)
        rc = proc.wait()
        if rc != 0:
            print(f"Graph {g} run exited with non-zero code {rc}")
            # Continue to next graph rather than aborting whole node run

    print("\nAll requested graphs submitted on this node. Check logs/ and output/ for progress.")


if __name__ == '__main__':
    main()
