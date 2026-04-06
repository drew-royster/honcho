#!/usr/bin/env python3
"""Run the Honcho API and deriver in a single container.

This keeps stage-1 consolidation simple: one app container runs both
processes, while Postgres, Redis, and embeddings remain separate services.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _start_process(name: str, argv: list[str]) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"Starting {name}: {' '.join(argv)}", flush=True)
    return subprocess.Popen(argv, env=env)


def main() -> int:
    api = _start_process(
        "api",
        ["/app/.venv/bin/fastapi", "run", "--host", "0.0.0.0", "src/main.py"],
    )
    deriver = _start_process(
        "deriver",
        ["/app/.venv/bin/python", "-m", "src.deriver"],
    )
    children = {"api": api, "deriver": deriver}

    shutting_down = False

    def _shutdown(signum: int, _frame) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print(f"Received signal {signum}; stopping child processes", flush=True)
        for proc in children.values():
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            for name, proc in children.items():
                code = proc.poll()
                if code is None:
                    continue
                if not shutting_down:
                    print(
                        f"{name} exited unexpectedly with code {code}; stopping remaining processes",
                        flush=True,
                    )
                    shutting_down = True
                    for other_name, other_proc in children.items():
                        if other_name != name and other_proc.poll() is None:
                            other_proc.terminate()
                else:
                    print(f"{name} exited with code {code}", flush=True)

                deadline = time.time() + 10
                for other_proc in children.values():
                    if other_proc is proc:
                        continue
                    while other_proc.poll() is None and time.time() < deadline:
                        time.sleep(0.1)
                    if other_proc.poll() is None:
                        other_proc.kill()

                return code if code is not None else 1

            time.sleep(0.5)
    finally:
        for proc in children.values():
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
