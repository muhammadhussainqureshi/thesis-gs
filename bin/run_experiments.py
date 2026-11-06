#!/usr/bin/env python3
"""Run a list of experiments from a JSON config, one after the other, with one log file per run.

    python bin/run_experiments.py configs/final_part_a.json --out outputs/final_a --data_dir data

A failing run is logged and the next one still runs. Any other arguments are appended to every run
(e.g. `--quick` for a smoke test).
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--volumes", default=None, help="folder with 3D volumes; replaces VOLUMES in the config")
    ap.add_argument("--only", nargs="*", help="run only these run names")
    a, extra = ap.parse_known_args()          # anything unknown (e.g. --quick) is passed on to every run
    extra = [x for x in extra if x != "--"]
    cfg = json.load(open(a.config))
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    status = {}
    for run in cfg["runs"]:
        name = run["name"]
        if a.only and name not in a.only:
            continue
        args = list(run["args"])
        if "VOLUMES" in args:
            if not a.volumes:
                print(f"===> {name}: SKIPPED (needs --volumes)", flush=True)
                status[name] = {"ok": False, "skipped": "no --volumes"}
                continue
            args = [a.volumes if x == "VOLUMES" else x for x in args]
        cmd = [sys.executable, os.path.join(HERE, "thesis-gs")] + args + \
              ["--data_dir", a.data_dir, "--out", os.path.join(a.out, name)] + extra
        log_path = os.path.join(a.out, "logs", f"{name}.log")
        print(f"\n===== {name}: {' '.join(args)}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as logf:
            logf.write(" ".join(cmd) + "\n\n")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                print(line, end="", flush=True)
                logf.write(line)
            p.wait()
            logf.write(f"\nEXIT CODE {p.returncode}\n")
        status[name] = {"ok": p.returncode == 0, "minutes": round((time.time() - t0) / 60, 1)}
        print(f"===> {name}: {'OK' if p.returncode == 0 else 'FAILED, see ' + log_path}", flush=True)
    json.dump(status, open(os.path.join(a.out, "status.json"), "w"), indent=1)
    print("\nSummary:", json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
