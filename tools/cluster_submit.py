"""
Cluster submission helper to launch runs from a head node across multiple fatanodes.
- Splits the full graph set across nodes (excluding random-lobster by default).
- For each node, SSHes in and runs tools/run_on_node.py with nohup so it survives disconnects.
- Uses a shared project directory (recommended on fatanode-data) so all nodes write to the same output/ tree.

Example (from fatanode-head):
  python3 tools/cluster_submit.py \
    --nodes fatanode01,fatanode02,fatanode03,fatanode04,fatanode05 \
    --project-dir /mnt/fatanode-data/CostFnFire \
    --size 100 --trials 50 --budgets 1,2,3,4,5 \
    --jobs-per-process 12 --max-procs 8 --timeout 7200 --resume

Notes:
- Tie-break heuristics: do NOT pass --heuristics to include defaults with tie-breakers (Degree/Threat/Cost combos).
- Costs: do NOT pass --costs to include the default 5 cost functions from simulation.py.
- Logs: per-node stdout/stderr go to <project-dir>/<log-dir>/<exp-id>/node-<host>.{log,err} (default log-dir is logs/cluster).
- Summary: once all nodes finish, run tools/summarize_results.py --base output/<exp-id> to aggregate.
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

DEFAULT_GRAPHS = [
    "erdos-renyi",
    "barabasi-albert",
    "watts-strogatz",
    "powerlaw-cluster",
    "scale-free",
    "random geometric",
    "random n-regular",
    "connected-caveman",
]


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Submit CostFnFire experiments across multiple nodes via SSH")
    ap.add_argument("--nodes", required=True, help="Comma-separated hostnames (e.g., fatanode01,fatanode02)")
    ap.add_argument("--project-dir", required=True,
                    help="Absolute path to the shared project dir on all nodes (e.g., /mnt/fatanode-data/CostFnFire)")
    ap.add_argument("--graphs", default=','.join(DEFAULT_GRAPHS),
                    help="Comma-separated graphs to split across nodes; default is the full set (no lobster)")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--budgets", type=str, default="1,2,3,4,5")
    ap.add_argument("--outbreak", type=str, default="rand")
    ap.add_argument("--jobs-per-process", type=int, default=12)
    ap.add_argument("--max-procs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=0,
                    help="Kill a simulation process if it exceeds this many seconds (0 = no timeout)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip launching a param if its results appear to already exist (resume mode)")
    ap.add_argument("--save-details", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--exp-id", type=str, default=None,
                    help="Experiment id used across all nodes; default is timestamp")
    ap.add_argument("--python", type=str, default="python3", help="Python executable to use on nodes")
    ap.add_argument("--log-dir", type=str, default="logs/cluster",
                    help="Log directory relative to --project-dir or absolute; default: logs/cluster")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def chunk(lst: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [lst]
    out = [[] for _ in range(min(n, len(lst)))]
    for i, item in enumerate(lst):
        out[i % len(out)].append(item)
    return out


def main():
    args = parse_args(sys.argv[1:])
    nodes = [h.strip() for h in args.nodes.split(',') if h.strip()]
    graphs = [g.strip() for g in args.graphs.split(',') if g.strip()]
    exp_id = args.exp_id or datetime.now().strftime('%Y%m%d%H%M%S')

    proj = Path(args.project_dir)
    if not proj.is_absolute():
        print("--project-dir must be an absolute path shared across nodes", file=sys.stderr)
        sys.exit(2)

    # Compute the log directory path relative to the project dir unless absolute provided.
    user_log_dir = Path(args.log_dir)
    if user_log_dir.is_absolute():
        # Use absolute log dir as-is on each node
        remote_log_dir = user_log_dir / exp_id
        log_dir_display = remote_log_dir
        ensure_prefix = shlex.quote(str(remote_log_dir))
        redir_prefix = shlex.quote(str(remote_log_dir))
        mkdir_cmd = f"mkdir -p {ensure_prefix}"
    else:
        # Store under project dir
        remote_log_dir = user_log_dir / exp_id  # relative path
        log_dir_display = proj / remote_log_dir
        ensure_prefix = shlex.quote(str(remote_log_dir))
        redir_prefix = shlex.quote(str(remote_log_dir))
        mkdir_cmd = f"mkdir -p {ensure_prefix}"

    # Split graphs across nodes, round-robin
    graph_chunks = chunk(graphs, len(nodes))

    print(f"Submitting exp-id={exp_id} across {len(nodes)} node(s)")
    print(f"Project dir: {proj}")
    print(f"Log dir: {log_dir_display}")
    print(f"Graph split: {graph_chunks}")

    for host, graphs_for_host in zip(nodes, graph_chunks):
        if not graphs_for_host:
            continue
        graphs_arg = ','.join(graphs_for_host)

        # Build a robust bootstrap: prefer venv python if available; try to create it; fall back to system python
        py = shlex.quote(args.python)
        proj_quoted = shlex.quote(str(proj))
        graphs_quoted = shlex.quote(graphs_arg)
        budgets_quoted = shlex.quote(args.budgets)
        outbreak_quoted = shlex.quote(args.outbreak)
        exp_id_quoted = shlex.quote(exp_id)

        # Remote script avoids `activate` and uses explicit python path
        remote_script = (
            f"cd {proj_quoted} ; "
            # ensure log dir exists (relative to project dir)
            f"{mkdir_cmd} ; "
            # choose python
            "if [ -x .venv/bin/python ]; then PYCMD=.venv/bin/python; "
            f"else {py} -m venv .venv >/dev/null 2>&1 || true; "
            "     if [ -x .venv/bin/python ]; then PYCMD=.venv/bin/python; else PYCMD=\"" + shlex.quote(args.python) + "\"; fi; fi; "
            # install requirements with chosen python; if that fails (no perms), fallback to --user on system python
            "${PYCMD} -m pip install -q -r requirements.txt || "
            f"{py} -m pip install -q --user -r requirements.txt ; "
            # launch
            "nohup ${PYCMD} tools/run_on_node.py "
            f"--graphs {graphs_quoted} "
            f"--size {args.size} --trials {args.trials} --budgets {budgets_quoted} "
            f"--outbreak {outbreak_quoted} "
            f"--jobs-per-process {args.jobs_per_process} --max-procs {args.max_procs} "
            f"--output-dir output --exp-id {exp_id_quoted} "
            + (" --resume" if args.resume else "")
            + (" --save-details" if args.save_details else "")
            + (" --progress" if args.progress else "")
            + (f" --timeout {args.timeout}" if args.timeout else "")
            + f" > {redir_prefix}/node-{host}.log 2> {redir_prefix}/node-{host}.err &"
        )

        ssh_cmd = ["ssh", host, remote_script]

        print("\n=== Host:", host)
        print("Graphs:", graphs_for_host)
        print("CMD:", ' '.join(shlex.quote(c) for c in ssh_cmd))
        if args.dry_run:
            continue
        try:
            p = subprocess.Popen(ssh_cmd)
            rc = p.wait()
            if rc != 0:
                print(f"Warning: SSH to {host} exited with code {rc}")
        except Exception as e:
            print(f"Error launching on {host}: {e}")

    print("\nAll submissions issued. Tail per-node logs under:")
    print(log_dir_display)
    print("Example:")
    print(f"  tail -f {log_dir_display}/node-{nodes[0]}.log")
    print("When finished, summarise with:")
    print(f"  {args.python} tools/summarize_results.py --base output/{exp_id}")


def main_wrapper():
    main()


if __name__ == "__main__":
    main_wrapper()
