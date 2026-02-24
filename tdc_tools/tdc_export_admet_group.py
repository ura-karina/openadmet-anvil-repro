from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _group_dataset_dir(group: Any, dataset_name: str) -> Path:
    base = Path(getattr(group, "path", "./data"))
    return base / dataset_name


def _load_test_split(group: Any, dataset_name: str) -> pd.DataFrame:
    if hasattr(group, "get_test"):
        return group.get_test(dataset_name)  # type: ignore[arg-type]
    ds_dir = _group_dataset_dir(group, dataset_name)
    file_format = getattr(group, "file_format", "csv")
    if file_format == "pkl":
        path = ds_dir / "test.pkl"
        if path.exists():
            return pd.read_pickle(path)
    path = ds_dir / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test split for {dataset_name} at {path}")
    return pd.read_csv(path)


def _train_valid_split(group: Any, dataset_name: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        # Newer API: (benchmark, seed=seed)
        return group.get_train_valid_split(dataset_name, seed=seed)  # type: ignore[misc]
    except TypeError:
        # Older API: (seed, benchmark)
        return group.get_train_valid_split(seed, dataset_name)  # type: ignore[misc]


@dataclass(frozen=True)
class ExportResult:
    dataset_name: str
    task_type: str
    n_test: int
    seeds: list[int]
    metric: str | None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_split_df(df: pd.DataFrame) -> pd.DataFrame:
    # TDC typically uses columns Drug + Y. Be robust to minor variations.
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("drug", "smiles", "canonical_smiles"):
            col_map[c] = "OPENADMET_SMILES"
        if lc in ("y", "label", "labels"):
            col_map[c] = "y"
    df = df.rename(columns=col_map)

    if "OPENADMET_SMILES" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    out = df[["OPENADMET_SMILES", "y"]].copy()
    out = out.dropna(subset=["OPENADMET_SMILES", "y"]).reset_index(drop=True)
    return out


def export_admet_group(out_root: Path, seeds: list[int]) -> list[ExportResult]:
    try:
        from tdc.benchmark_group import BenchmarkGroup  # type: ignore
    except Exception:
        from tdc.benchmark_group.base_group import BenchmarkGroup  # type: ignore

    _ensure_dir(out_root)

    group = BenchmarkGroup(name="ADMET_Group")

    # Discover dataset names.
    dataset_names: list[str] = []
    for attr in ("dataset_names", "datasets", "benchmark_names", "name_list"):
        if hasattr(group, attr):
            val = getattr(group, attr)
            dataset_names = list(val() if callable(val) else val)
            if dataset_names:
                break
    if not dataset_names:
        # Fallback: try internal dict
        maybe = getattr(group, "datasets", None)
        if isinstance(maybe, dict):
            dataset_names = sorted(maybe.keys())
    if not dataset_names:
        raise RuntimeError("Failed to discover ADMET_Group dataset names from BenchmarkGroup")

    results: list[ExportResult] = []

    for dataset_name in dataset_names:
        ds_dir = out_root / dataset_name
        _ensure_dir(ds_dir)

        # Fixed test set for the benchmark.
        test_df = _load_test_split(group, dataset_name)
        test_df = _normalize_split_df(test_df)
        test_path = ds_dir / "test.parquet"
        test_df.to_parquet(test_path, index=False)

        # Task type if available.
        task_type = "unknown"
        metric: str | None = None
        try:
            info: Any = group.get_info(dataset_name)  # type: ignore[arg-type]
            if isinstance(info, dict):
                task_type = str(info.get("task", info.get("task_type", task_type)))
                raw_metric = info.get("metric", info.get("metric_name"))
                if raw_metric is not None:
                    metric = str(raw_metric)
        except Exception:
            pass

        # Per-seed train/valid splits (TDC protocol).
        for seed in seeds:
            seed_dir = ds_dir / f"seed_{seed}"
            _ensure_dir(seed_dir)
            train_df, valid_df = _train_valid_split(group, dataset_name, seed)
            train_df = _normalize_split_df(train_df)
            valid_df = _normalize_split_df(valid_df)
            train_df.to_parquet(seed_dir / "train.parquet", index=False)
            valid_df.to_parquet(seed_dir / "valid.parquet", index=False)
            trainval_df = pd.concat([train_df, valid_df], ignore_index=True)
            trainval_df.to_parquet(seed_dir / "trainval.parquet", index=False)

        meta = {
            "dataset_name": dataset_name,
            "task_type": task_type,
            "metric": metric,
            "n_test": int(len(test_df)),
            "seeds": seeds,
        }
        (ds_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        results.append(
            ExportResult(
                dataset_name=dataset_name,
                task_type=task_type,
                n_test=int(len(test_df)),
                seeds=seeds,
                metric=metric,
            )
        )

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    out_root: Path = args.out_root
    seeds: list[int] = args.seeds

    # Prevent surprising writes elsewhere if someone passes a relative path.
    out_root = out_root.expanduser().resolve()
    os.makedirs(out_root, exist_ok=True)

    results = export_admet_group(out_root=out_root, seeds=seeds)
    print(f"Exported {len(results)} datasets to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
