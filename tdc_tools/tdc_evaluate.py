from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class Row:
    dataset: str
    task_kind: str
    model_type: str
    feat_type: str
    hp_id: str
    n_seeds: int
    metric_primary: str
    metric_secondary: str
    metric_primary_higher_is_better: bool
    metric_primary_mean: float
    metric_primary_std: float
    metric_secondary_mean: float | None
    metric_secondary_std: float | None


def _discover_task_kind(meta_path: Path) -> str:
    if not meta_path.exists():
        return "unknown"
    meta = json.loads(meta_path.read_text())
    raw = str(meta.get("task_type") or meta.get("task") or "").lower()
    if "class" in raw:
        return "classification"
    if "regress" in raw:
        return "regression"
    return "unknown"


def _load_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _find_seed_dirs(run_tag_dir: Path) -> list[Path]:
    # run_tag_dir = runs_root / tag
    return sorted([p for p in run_tag_dir.glob("*/") if p.is_dir()])


def _extract_recipe_meta(run_dir: Path) -> dict[str, Any]:
    recipe_path = run_dir / "anvil_recipe.yaml"
    if not recipe_path.exists():
        return {}
    import yaml

    return yaml.safe_load(recipe_path.read_text())


def _predict(run_dir: Path, test_parquet: Path, input_col: str, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "openadmet",
        "predict",
        "--input-path",
        str(test_parquet),
        "--input-col",
        input_col,
        "--model-dir",
        str(run_dir),
        "--output-csv",
        str(out_csv),
        "--accelerator",
        "cpu",
    ]
    subprocess.run(cmd, check=True)
    return out_csv


def _metric_alias(metric: str) -> str:
    raw = metric.strip().lower()
    aliases = {
        "auc": "roc_auc",
        "auroc": "roc_auc",
        "roc_auc": "roc_auc",
        "auprc": "pr_auc",
        "prauc": "pr_auc",
        "ap": "pr_auc",
        "pr_auc": "pr_auc",
        "mae": "mae",
        "rmse": "rmse",
        "mse": "mse",
        "spearman": "spearmanr",
        "spearmanr": "spearmanr",
        "pearson": "pearsonr",
        "pearsonr": "pearsonr",
        "r2": "r2",
    }
    return aliases.get(raw, raw)


def _metric_from_meta(meta: dict[str, Any]) -> str | None:
    raw = meta.get("metric")
    if raw is None:
        raw = meta.get("metrics")
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if raw is None:
        return None
    return _metric_alias(str(raw))


def _primary_secondary_metrics(task_kind: str, meta: dict[str, Any]):
    meta_metric = _metric_from_meta(meta)
    if task_kind == "classification":
        primary = meta_metric if meta_metric in ("roc_auc", "pr_auc") else "roc_auc"
        secondary = "pr_auc" if primary == "roc_auc" else "roc_auc"
        return (primary, secondary)
    primary = meta_metric if meta_metric in ("mae", "spearmanr", "rmse", "mse", "pearsonr", "r2") else "mae"
    secondary = "spearmanr" if primary != "spearmanr" else "mae"
    return (primary, secondary)


def _compute_metrics(task_kind: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if task_kind == "classification":
        # Expect y_pred as probability of positive class.
        return {
            "roc_auc": float(roc_auc_score(y_true, y_pred)),
            "pr_auc": float(average_precision_score(y_true, y_pred)),
        }
    # Regression
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    try:
        pear = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    except Exception:
        pear = float("nan")
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(math.sqrt(mse))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearmanr": float(rho),
        "pearsonr": float(pear),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _metric_higher_is_better(metric: str) -> bool:
    return metric in {"roc_auc", "pr_auc", "spearmanr", "pearsonr", "r2"}


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) > 1:
        return float(arr.mean()), float(arr.std(ddof=1))
    return float(arr.mean()), float("nan")


def _parse_prob(val: Any) -> Any:
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            return val
        if isinstance(parsed, (list, tuple)) and parsed:
            if len(parsed) == 2:
                return float(parsed[1])
            return float(parsed[-1])
        return parsed
    return val


def _coerce_pred(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        series = series.apply(_parse_prob)
    arr = series.to_numpy()
    if arr.ndim > 1 and arr.shape[1] > 1:
        return np.asarray(arr[:, -1], dtype=float)
    return np.asarray(arr, dtype=float)


def build_leaderboard(data_root: Path, runs_root: Path, out_csv: Path) -> None:
    data_root = data_root.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []

    # runs_root layout from tdc_run_anvil_matrix.py: runs_root/<tag>/<recipe_hash>/
    for tag_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        # Parse tag: tdc__<dataset>__seed<seed>__<model_type>__<feat_id>__<hp_id>
        parts = tag_dir.name.split("__")
        if len(parts) < 6 or parts[0] not in {"tdc", "tdc_ens"}:
            continue
        dataset = parts[1]
        seed_str = parts[2]
        model_type = parts[3]
        feat_type = parts[4]
        hp_id_from_tag = parts[5]

        dataset_dir = data_root / dataset
        dataset_meta = _load_meta(dataset_dir / "meta.json")
        task_kind = _discover_task_kind(dataset_dir / "meta.json")
        if task_kind == "unknown":
            # Default: infer from model name.
            task_kind = "classification" if "Classifier" in model_type else "regression"

        primary, secondary = _primary_secondary_metrics(task_kind, dataset_meta)

        # Each tag_dir corresponds to one seed; but there can be multiple hp_id via different tags in practice.
        for run_dir in _find_seed_dirs(tag_dir):
            _ = _extract_recipe_meta(run_dir)
            hp_id = hp_id_from_tag or "unknown"

            test_parquet = dataset_dir / "test.parquet"
            if not test_parquet.exists():
                continue

            # Load truth
            test_df = pd.read_parquet(test_parquet)
            if "y" not in test_df.columns:
                continue
            y_true = test_df["y"].to_numpy()

            # Run predict (cache per run_dir)
            pred_csv = run_dir / "tdc_test_predictions.csv"
            if not pred_csv.exists():
                _predict(run_dir=run_dir, test_parquet=test_parquet, input_col="OPENADMET_SMILES", out_csv=pred_csv)

            pred_df = pd.read_csv(pred_csv)

            # Find the prediction column for this run tag.
            # openadmet predict writes OADMET_PRED_<metadata.tag>_<taskname>
            pred_cols = [c for c in pred_df.columns if c.startswith("OADMET_PRED_") and c.endswith("_y")]
            if not pred_cols:
                # fallback: any OADMET_PRED column
                pred_cols = [c for c in pred_df.columns if c.startswith("OADMET_PRED_")]
            if not pred_cols:
                continue

            # For classification, ensure we take positive class probability if a 2-col array got stringified.
            y_pred = _coerce_pred(pred_df[pred_cols[0]])

            # Basic label normalization for classification.
            if task_kind == "classification":
                y_true_norm = np.asarray(y_true, dtype=float)
                # Map {-1,1} -> {0,1}
                uniq = set(np.unique(y_true_norm).tolist())
                if uniq == {-1.0, 1.0}:
                    y_true_norm = (y_true_norm > 0).astype(int)
                else:
                    y_true_norm = y_true_norm.astype(int)
                y_true_use = y_true_norm
            else:
                y_true_use = np.asarray(y_true, dtype=float)

            metrics = _compute_metrics(task_kind, y_true_use, y_pred)
            metric_primary_higher_is_better = _metric_higher_is_better(primary)

            # Store one row per (dataset, model, feat, hp_id, seed-run). We aggregate below.
            rows.append(
                Row(
                    dataset=dataset,
                    task_kind=task_kind,
                    model_type=model_type,
                    feat_type=feat_type,
                    hp_id=hp_id,
                    n_seeds=1,
                    metric_primary=primary,
                    metric_secondary=secondary,
                    metric_primary_higher_is_better=metric_primary_higher_is_better,
                    metric_primary_mean=float(metrics[primary]),
                    metric_primary_std=float("nan"),
                    metric_secondary_mean=float(metrics[secondary]),
                    metric_secondary_std=float("nan"),
                )
            )

    # Aggregate to dataset/model/feat/hp_id.
    grouped: dict[tuple[str, str, str, str, str], list[Row]] = {}
    for r in rows:
        key = (r.dataset, r.task_kind, r.model_type, r.feat_type, r.hp_id, r.metric_primary, r.metric_secondary, r.metric_primary_higher_is_better)
        grouped.setdefault(key, []).append(r)

    out_rows: list[dict[str, Any]] = []
    for (dataset, task_kind, model_type, feat_type, hp_id, primary, secondary, higher_is_better), rs in sorted(grouped.items()):
        prim_vals = [x.metric_primary_mean for x in rs]
        sec_vals = [x.metric_secondary_mean for x in rs if x.metric_secondary_mean is not None]

        prim_mean, prim_std = _mean_std(prim_vals)
        sec_mean, sec_std = _mean_std(sec_vals) if sec_vals else (math.nan, math.nan)

        out_rows.append(
            {
                "dataset": dataset,
                "task_kind": task_kind,
                "model_type": model_type,
                "feat_type": feat_type,
                "hp_id": hp_id,
                "n_seeds": len(rs),
                "metric_primary": primary,
                "metric_secondary": secondary,
                "metric_primary_higher_is_better": higher_is_better,
                f"{primary}_mean": prim_mean,
                f"{primary}_std": prim_std,
                f"{secondary}_mean": sec_mean,
                f"{secondary}_std": sec_std,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(out_rows[0].keys()) if out_rows else ["dataset"])
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    build_leaderboard(data_root=args.data_root, runs_root=args.runs_root, out_csv=args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
