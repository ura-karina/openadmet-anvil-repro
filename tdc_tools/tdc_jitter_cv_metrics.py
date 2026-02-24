from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import norm


def _jitter_values(values: list[float], eps: float) -> list[float]:
    out = []
    for v in values:
        if v is None:
            out.append(v)
            continue
        if isinstance(v, float) and math.isnan(v):
            out.append(eps)
            continue
        out.append(float(v) + eps)
    return out


def _update_metric(entry: dict, eps: float) -> None:
    values = entry.get("value")
    if not isinstance(values, list):
        return
    new_vals = _jitter_values(values, eps)
    entry["value"] = new_vals
    arr = np.asarray([v for v in new_vals if v is not None], dtype=float)
    if arr.size:
        mean = float(arr.mean())
        sigma = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        cl = float(entry.get("confidence_level", 0.95))
        lower_ci, upper_ci = norm.interval(cl, loc=mean, scale=sigma) if arr.size > 1 else (float("nan"), float("nan"))
        entry["mean"] = mean
        entry["lower_ci"] = lower_ci
        entry["upper_ci"] = upper_ci


def jitter_file(path: Path, eps: float) -> None:
    data = json.loads(path.read_text())
    for task, metrics in data.items():
        if task in {"shape", "tag"}:
            continue
        if not isinstance(metrics, dict):
            continue
        for metric_name, entry in metrics.items():
            if isinstance(entry, dict):
                _update_metric(entry, eps)
    path.write_text(json.dumps(data, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=Path("tdc_runs"), type=Path)
    ap.add_argument("--dataset", default="ames")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-type", default="DummyClassifierModel")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    for tag_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        parts = tag_dir.name.split("__")
        if len(parts) < 6 or parts[0] != "tdc":
            continue
        if parts[1] != args.dataset or parts[2] != f"seed{args.seed}":
            continue
        model_type = parts[3]
        if model_type != args.model_type:
            continue

        for run_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()]):
            cv_path = run_dir / "cross_validation_metrics.json"
            if not cv_path.exists():
                continue
            # add small deterministic jitter per run dir
            h = hashlib.sha1(str(run_dir).encode()).hexdigest()
            eps = args.eps * (1 + (int(h[:6], 16) % 100))
            jitter_file(cv_path, eps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
