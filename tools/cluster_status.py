"""
Quick status checker for multi-node runs launched via tools/cluster_submit.py.
- Shows per-node process counts (run_on_node/launch_batch/simulation) via SSH.
- Tails the last few lines of the per-node log files written by cluster_submit.

Example:
  python3 tools/cluster_status.py \
    --nodes fatanode-01,fatanode-02,fatanode-03,fatanode-04,fatanode-05 \
    --project-dir /users/grad/ekelly/CostFnRepo/CostFnFire \
    --exp-id 20250903120116

If --exp-id is omitted, the newest directory under <project-dir>/logs/cluster is used.
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Show per-node run status and tail logs")
    ap.add_argument("--nodes", required=True, help="Comma-separated hostnames")
    ap.add_argument("--project-dir", required=True, help="Absolute path to project dir shared by nodes")
    ap.add_argument("--exp-id", default=None, help="Experiment id; default is latest under logs/cluster")
    ap.add_argument("--log-dir", default="logs/cluster", help="Relative log dir under project-dir or absolute path")
    ap.add_argument("--tail", type=int, default=5, help="How many log lines to show per node")
    return ap.parse_args(argv)


def latest_expid(log_root: Path) -> Optional[str]:
    if not log_root.exists():
        return None
    try:
        candidates = [p.name for p in log_root.iterdir() if p.is_dir()]
        if not candidates:
            return None
        return sorted(candidates)[-1]
    except Exception:
        return None


def ssh_run(host: str, cmd: str) -> str:
    try:
        out = subprocess.check_output(["ssh", host, cmd], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except subprocess.CalledProcessError:
        return ""


def count_procs(host: str, pattern: str) -> int:
    cmd = f"pgrep -af {pattern} | wc -l"
    out = ssh_run(host, cmd)
    try:
        return int(out)
    except Exception:
        return 0


def main():
    args = parse_args()
    nodes = [n.strip() for n in args.nodes.split(',') if n.strip()]

    proj = Path(args.project_dir)
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_root = proj / log_dir
    else:
        log_root = log_dir

    exp_id = args.exp_id or latest_expid(log_root)
    if not exp_id:
        print(f"No exp-id provided and none found under {log_root}")
        return

    print(f"Status for exp-id={exp_id}\nLog root: {log_root / exp_id}\n")

    for host in nodes:
        ron = count_procs(host, "tools/run_on_node.py")
        lba = count_procs(host, "tools/launch_batch.py")
        sim = count_procs(host, "simulation.py")
        print(f"=== {host}")
        print(f"  procs: run_on_node={ron} launch_batch={lba} simulation={sim}")
        log_path = (log_root / exp_id / f"node-{host}.log")
        err_path = (log_root / exp_id / f"node-{host}.err")
        try:
            if log_path.exists():
                lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-args.tail:]
                print("  log tail:")
                for ln in lines:
                    print("   ", ln)
            else:
                print(f"  log missing: {log_path}")
        except Exception as e:
            print(f"  log read error: {e}")
        try:
            if err_path.exists():
                lines = err_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-args.tail:]
                if lines:
                    print("  err tail:")
                    for ln in lines:
                        print("   ", ln)
            else:
                print(f"  err missing: {err_path}")
        except Exception as e:
            print(f"  err read error: {e}")
        print()


if __name__ == "__main__":
    main()

