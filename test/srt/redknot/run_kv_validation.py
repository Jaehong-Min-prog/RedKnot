#!/usr/bin/env python3
"""Run the published KV integration regression suite without a checkpoint.

CPU mode hides CUDA. CUDA mode uses the caller's CUDA_VISIBLE_DEVICES and
requires two available devices to execute every case. This runner does not
start, stop, or manage any GPU jobs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--junitxml", type=Path)
    parser.add_argument("--list-tests", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    manifest = Path(__file__).with_name("kv_validation_tests.txt")
    tests = [line.strip() for line in manifest.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not tests or any(not (root / path).is_file() for path in tests):
        parser.error("the published validation manifest is incomplete")
    if args.list_tests:
        print("\n".join(tests))
        return 0
    if args.mode == "cuda":
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            parser.error("CUDA mode needs two visible GPUs; set CUDA_VISIBLE_DEVICES to the allocated devices")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    if args.mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", *tests]
    if args.junitxml:
        command.append("--junitxml=" + str(args.junitxml.resolve()))
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
