from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


def _dataset_names(data_root: Path) -> list[str]:
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])


def _try_call(obj: Any, name: str, dataset: str) -> Any | None:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(dataset)
    except TypeError:
        try:
            return fn(dataset_name=dataset)
        except Exception:
            return None
    except Exception:
        return None


def _best_row(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    lower_cols = {c.lower(): c for c in df.columns}

    metric_col = None
    value_col = None
    model_col = None

    for key in ("metric", "metric_name", "metric_type"):
        if key in lower_cols:
            metric_col = lower_cols[key]
            break
    for key in ("value", "score", "performance", "result"):
        if key in lower_cols:
            value_col = lower_cols[key]
            break
    for key in ("model", "method", "model_name", "baseline"):
        if key in lower_cols:
            model_col = lower_cols[key]
            break

    # If no explicit value column, try to find the first numeric column.
    if value_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break

    if value_col is None:
        return None

    metric = str(df[metric_col].iloc[0]) if metric_col else "unknown"
    metric_lower = metric.lower()
    higher_is_better = metric_lower in {"roc_auc", "auc", "auroc", "pr_auc", "auprc", "ap", "accuracy", "r2", "spearman", "spearmanr", "pearson", "pearsonr"}

    best_idx = df[value_col].astype(float).idxmax() if higher_is_better else df[value_col].astype(float).idxmin()
    row = df.loc[best_idx]
    return {
        "metric": metric,
        "value": float(row[value_col]),
        "model": str(row[model_col]) if model_col and model_col in row else "unknown",
    }


def fetch_baselines(data_root: Path, out_csv: Path) -> None:
    data_root = data_root.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    datasets = _dataset_names(data_root)
    records: list[dict[str, Any]] = []

    try:
        try:
            from tdc.benchmark_group import BenchmarkGroup  # type: ignore
        except Exception:
            from tdc.benchmark_group.base_group import BenchmarkGroup  # type: ignore

        group = BenchmarkGroup(name="ADMET_Group")
    except Exception:
        group = None

    for dataset in datasets:
        record: dict[str, Any] = {
            "dataset": dataset,
            "metric": "unknown",
            "value": "",
            "model": "",
            "source": "tdc",
        }

        if group is None:
            record["source"] = "tdc_unavailable"
            records.append(record)
            continue

        leaderboard = None
        for method in ("get_leaderboard", "get_results", "get_benchmark_results", "get_leaderboard_results"):
            leaderboard = _try_call(group, method, dataset)
            if leaderboard is not None:
                break

        if leaderboard is None:
            record["source"] = "tdc_missing"
            records.append(record)
            continue

        try:
            df = pd.DataFrame(leaderboard)
        except Exception:
            record["source"] = "tdc_unparseable"
            records.append(record)
            continue

        best = _best_row(df)
        if best is None:
            record["source"] = "tdc_no_numeric"
            records.append(record)
            continue

        record.update(best)
        records.append(record)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "value", "model", "source"])
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    fetch_baselines(data_root=args.data_root, out_csv=args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
