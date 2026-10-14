"""Conservative wall-clock GPU rental accounting; stopping a job does not stop EC2 billing."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def status(path=ROOT / "results/budget.json"):
    ledger = json.loads(Path(path).read_text())
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(ledger["started_utc"])).total_seconds()
    cost = ledger["prior_cost_usd"] + max(0, elapsed) / 3600 * ledger["hourly_usd"]
    return {**ledger, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_hours": elapsed / 3600, "estimated_cumulative_usd": cost,
            "remaining_usd": ledger["cap_usd"] - cost,
            "may_run": cost < ledger["stop_work_at_usd"] and elapsed < ledger["max_hours"] * 3600}


def require_budget():
    s = status()
    if not s["may_run"]:
        raise RuntimeError("GPU work budget exhausted. EC2 must be stopped separately to stop billing.")
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    s = status()
    print(json.dumps(s, indent=2))
    if args.check and not s["may_run"]:
        raise SystemExit(2)
