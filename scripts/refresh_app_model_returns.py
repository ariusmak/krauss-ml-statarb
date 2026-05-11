"""Background-safe app refresh wrapper.

Streamlit starts this script in a detached process when a browser session opens.
The wrapper owns the lock file, runs the incremental model-return refresh, and
always releases the lock when the child command exits.  The child command also
refreshes ``app/data/live_latest_positions.parquet`` so Streamlit never needs
to import the backend simulator during page render.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DATA = ROOT / "app" / "data"
LOCK_PATH = APP_DATA / ".model_returns_refresh.lock"
LOCK_TTL_SECONDS = 60 * 60


def _log(message: str) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{now}] {message}", flush=True)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock_is_active() -> bool:
    if not LOCK_PATH.exists():
        return False
    try:
        payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False

    started_at = float(payload.get("started_at", 0.0))
    if time.time() - started_at > LOCK_TTL_SECONDS:
        return False

    pid = payload.get("pid")
    return not isinstance(pid, int) or _pid_is_alive(pid)


def main() -> int:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    if _lock_is_active():
        _log("another model_returns refresh is already running")
        return 0

    LOCK_PATH.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": time.time(),
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "scripts/build_model_returns.py", "--refresh-sources"]

    try:
        _log("starting incremental app refresh")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        _log("incremental app refresh complete")
    except subprocess.CalledProcessError as exc:
        _log(f"incremental app refresh failed with exit code {exc.returncode}")
        return exc.returncode
    finally:
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
