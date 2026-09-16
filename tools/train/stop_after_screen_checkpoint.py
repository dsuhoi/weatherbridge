#!/usr/bin/env python3
"""Stop one legacy long-running queue after its screen checkpoint is ready."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def _processes() -> dict[int, tuple[int, int, bytes]]:
    result: dict[int, tuple[int, int, bytes]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().split()
            result[int(entry.name)] = (
                int(stat[3]),
                int(stat[21]),
                (entry / "cmdline").read_bytes(),
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def _tree(
    root: int,
    processes: dict[int, tuple[int, int, bytes]],
) -> list[int]:
    descendants: list[int] = []
    changed = True
    while changed:
        changed = False
        for pid, (parent, _, _) in processes.items():
            if pid in descendants or pid == root:
                continue
            if parent == root or parent in descendants:
                descendants.append(pid)
                changed = True
    return [root, *descendants]


def _checkpoint_ready(
    python_bin: Path,
    status_script: Path,
    checkpoint: Path,
    min_epochs: int,
) -> bool:
    return (
        subprocess.run(
            [
                str(python_bin),
                str(status_script),
                str(checkpoint),
                "--min-epochs",
                str(min_epochs),
                "--quiet",
            ],
            check=False,
        ).returncode
        == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-pid", type=int, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--min-epochs", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/home/jovyan/.mlspace/envs/ai_scientist/bin/python"),
    )
    parser.add_argument(
        "--status-script",
        type=Path,
        default=Path(__file__).with_name("checkpoint_status.py"),
    )
    args = parser.parse_args()

    while not _checkpoint_ready(
        args.python_bin,
        args.status_script,
        args.checkpoint,
        args.min_epochs,
    ):
        if not Path(f"/proc/{args.queue_pid}").exists():
            raise SystemExit(f"queue pid {args.queue_pid} exited before checkpoint")
        time.sleep(args.poll_seconds)

    processes = _processes()
    tree = _tree(args.queue_pid, processes)
    marker = f"--exp_name\x00{args.experiment}\x00".encode()
    matching = [pid for pid in tree if marker in processes[pid][2]]
    if not matching:
        raise SystemExit(
            f"refusing to stop pid {args.queue_pid}: "
            f"experiment {args.experiment} is not in its process tree"
        )

    fingerprints = {
        pid: processes[pid][1]
        for pid in tree
    }
    for pid in tree:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(15.0)
    current = _processes()
    for pid, start_time in fingerprints.items():
        if pid not in current or current[pid][1] != start_time:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(
        f"stopped experiment={args.experiment} "
        f"after min_epochs={args.min_epochs}"
    )


if __name__ == "__main__":
    main()
