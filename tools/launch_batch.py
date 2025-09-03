"""
Launch multiple simulation.py processes on the current machine, one per parameter value, with per-process logs.
Designed for use on remote nodes. Processes are started in their own sessions and
stdout/stderr are redirected to log files so they continue after SSH disconnect.

Example:
  python3 tools/launch_batch.py \
    --graph "random geometric" --params 0.1,0.15,0.2 \
    --size 100 --trials 50 --budgets 1,2,3,4,5 \
    --costs "Uniform,Uniformly Random,Hesitancy-Binary" \
    --heuristics "Random,Degree,Threat,Cost" \
    --jobs-per-process 8 --max-procs 3 --output-dir output --exp-id RG-20250903 --timeout 3600 --resume

This will start up to 3 concurrent simulation processes, each using 8 workers internally, killing any that exceed 1 hour.
"""
import argparse
import shlex
import sys
import time
import subprocess
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import List


def parse_args(argv):
    p = argparse.ArgumentParser(description="Launch multiple simulation runs (one per parameter value)")
    p.add_argument("--graph", required=True, help="Graph type (e.g. 'erdos-renyi', 'barabasi-albert', 'random geometric', 'scale-free')")
    p.add_argument("--params", type=str, default=None, help="Comma-separated parameter values (e.g. '0.05,0.1' or '2,3,4'); if omitted, uses defaults")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--budgets", type=str, default="1,2,3,4,5")
    p.add_argument("--costs", type=str, default=None, help="Comma-separated cost function names (match simulation.py expectations); default is all")
    p.add_argument("--heuristics", type=str, default=None, help="Comma-separated heuristics (match simulation.py expectations); default is all")
    p.add_argument("--outbreak", type=str, default="rand", help="Outbreak node id or 'rand' (default)")
    p.add_argument("--jobs-per-process", type=int, default=1, help="--jobs passed to each simulation.py process")
    p.add_argument("--max-procs", type=int, default=1, help="Max concurrent simulation.py processes to run")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--exp-id", type=str, default=None, help="Experiment id (folder under output dir). Default: timestamp")
    p.add_argument("--save-details", action="store_true", help="Pass --save-details to simulation.py (heavy I/O)")
    p.add_argument("--progress", action="store_true", help="Pass --progress to simulation.py")
    p.add_argument("--timeout", type=int, default=0, help="Kill a simulation process if it exceeds this many seconds (0 = no timeout)")
    p.add_argument("--resume", action="store_true", help="Skip launching if results.csv for this param already exists")
    p.add_argument("--dry-run", action="store_true", help="Print commands only; do not launch")
    return p.parse_args(argv)


def default_params_for_graph(g: str):
    defaults = {
        'erdos-renyi': [0.05, 0.1, 0.15, 0.2, 0.25],
        'barabasi-albert': [1, 2, 3, 5, 8, 10],
        'watts-strogatz': [4, 6, 8, 10],
        'powerlaw-cluster': [2, 3, 4, 5],
        'scale-free': [-1],
        'random geometric': [0.05, 0.1, 0.15, 0.2, 0.25],
        'random n-regular': [2, 3, 4, 6, 8],
        'connected-caveman': [5, 10, 15, 20],
        # 'random-lobster': [-1],
    }
    return defaults.get(g, [-1])


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_command(param, args, exp_id: str) -> List[str]:
    base = [sys.executable, str((Path(__file__).parents[1] / 'simulation.py'))]
    base += ["--graph", args.graph, "--params", str(param), "--size", str(args.size), "--trials", str(args.trials)]
    base += ["--budgets", args.budgets, "--output-dir", args.output_dir, "--exp-id", exp_id]
    base += ["--jobs", str(args.jobs_per_process), "--outbreak", args.outbreak]
    if args.costs:
        base += ["--costs", args.costs]
    if args.heuristics:
        base += ["--heuristics", args.heuristics]
    if args.save_details:
        base += ["--save-details"]
    if args.progress:
        base += ["--progress"]
    return base


def result_file_exists(args, exp_id: str, param) -> bool:
    # Check for any results.csv under output/exp-id/<cost_fn>/<graph>/ containing this parameter string
    # Cheap check: just see if results.csv exists for this graph and any cost function
    base = Path(args.output_dir) / exp_id
    graph_dir_name = args.graph.replace(' ', '_')
    for cost_dir in base.iterdir() if base.exists() else []:
        candidate = cost_dir / graph_dir_name / 'results.csv'
        if candidate.exists():
            try:
                # Look for the param value in the parameter column to decide skip
                with candidate.open('r', encoding='utf-8', errors='ignore') as f:
                    if f",{param}," in f.read():
                        return True
            except Exception:
                continue
    return False


def kill_process_tree(proc: subprocess.Popen):
    try:
        # Kill the whole process group started with start_new_session=True
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def main():
    args = parse_args(sys.argv[1:])

    # Parse params
    if args.params:
        params = []
        for tok in args.params.split(','):
            tok = tok.strip()
            if tok == '':
                continue
            try:
                params.append(int(tok))
            except ValueError:
                try:
                    params.append(float(tok))
                except ValueError:
                    params.append(tok)
    else:
        params = default_params_for_graph(args.graph)

    exp_id = args.exp_id or datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = Path('logs') / exp_id / args.graph.replace(' ', '_')
    ensure_dir(log_dir)

    print(f"Launching {len(params)} process(es) for graph='{args.graph}' with exp-id={exp_id}")
    print(f"Logs: {log_dir}")

    procs: List[subprocess.Popen] = []
    meta = []  # track (proc, param, stdout, stderr, start_time)
    for param in params:
        if args.resume and result_file_exists(args, exp_id, param):
            print(f"Skipping param {param} because results already exist (resume mode).")
            continue
        # Throttle to max_procs
        while len(procs) >= args.max_procs:
            time.sleep(0.5)
            # Handle timeouts for running procs
            now = time.time()
            for i, (p, par, out_fp, err_fp, start_ts) in list(enumerate(meta)):
                if p.poll() is not None:
                    # Finished
                    try:
                        out_fp.close(); err_fp.close()
                    except Exception:
                        pass
                    print(f"Process {p.pid} for param={par} exited with code {p.returncode}")
                    procs.pop(i); meta.pop(i)
                    continue
                if args.timeout and (now - start_ts) > args.timeout:
                    print(f"Timeout reached for pid {p.pid} param={par}; killing...")
                    kill_process_tree(p)
                    try:
                        out_fp.write("\n[TIMEOUT] Killed by launcher due to --timeout.\n"); out_fp.flush()
                    except Exception:
                        pass
                    procs.pop(i); meta.pop(i)

        cmd = build_command(param, args, exp_id)
        log_path = log_dir / f"param_{str(param).replace(' ', '_').replace('/', '-')}.log"
        print("CMD:", shlex.join(cmd))
        if args.dry_run:
            continue
        stdout = open(log_path, 'a', buffering=1)
        stderr = open(log_path.with_suffix('.err.log'), 'a', buffering=1)
        # Start in a new session so it survives SIGHUP; redirect both stdout and stderr
        p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, start_new_session=True)
        procs.append(p)
        meta.append((p, param, stdout, stderr, time.time()))
        time.sleep(0.2)

    if args.dry_run:
        print("Dry run complete; no processes launched.")
        return

    if not procs:
        print("Nothing to run.")
        return

    print("All processes launched. PIDs:", [p.pid for p in procs])
    print("Waiting for processes to finish...")
    # Wait for completion and enforce timeout
    while procs:
        time.sleep(0.5)
        now = time.time()
        for i, (p, par, out_fp, err_fp, start_ts) in list(enumerate(meta)):
            if p.poll() is not None:
                try:
                    out_fp.close(); err_fp.close()
                except Exception:
                    pass
                print(f"Process {p.pid} for param={par} exited with code {p.returncode}")
                procs.pop(i); meta.pop(i)
                continue
            if args.timeout and (now - start_ts) > args.timeout:
                print(f"Timeout reached for pid {p.pid} param={par}; killing...")
                kill_process_tree(p)
                try:
                    out_fp.write("\n[TIMEOUT] Killed by launcher due to --timeout.\n"); out_fp.flush()
                except Exception:
                    pass
                procs.pop(i); meta.pop(i)

    # summarize results
    try:
        try:
            from tools.summarize_results import summarize
        except Exception:
            # Adjust sys.path so running from tools/ or project root both work
            sys.path.append(str(Path(__file__).parents[1]))
            from tools.summarize_results import summarize
        out_base = Path(args.output_dir) / exp_id
        summary_path = out_base / 'summary.csv'
        ensure_dir(out_base)
        rows = summarize(out_base)
        if rows:
            print(f"Wrote summary with {len(rows)-1} rows:", summary_path)
        else:
            print("No results found to summarise (yet).")
    except Exception as e:
        print("Summary step failed:", e)


if __name__ == "__main__":
    main()
