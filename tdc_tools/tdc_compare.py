from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

from openadmet.models.comparison.posthoc import PostHocComparison


def _iter_run_dirs(runs_root: Path, seed: int | None) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for tag_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        parts = tag_dir.name.split("__")
        if len(parts) < 6 or parts[0] != "tdc":
            continue
        dataset = parts[1]
        seed_part = parts[2]
        if seed is not None:
            try:
                seed_num = int(seed_part.replace("seed", ""))
            except ValueError:
                continue
            if seed_num != seed:
                continue

        for run_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()]):
            if not (run_dir / "anvil_recipe.yaml").exists():
                continue
            cv_files = list(run_dir.glob("cross_validation_metrics*.json"))
            if not cv_files:
                continue
            grouped.setdefault(dataset, []).append(run_dir)
    return grouped


def _flatten_args(flag: str, values: Iterable[str]) -> list[str]:
    args: list[str] = []
    for v in values:
        args.extend([flag, v])
    return args


def _configure_compare_metrics(task_type: str) -> None:
    if task_type == "classification":
        PostHocComparison._metrics_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
        PostHocComparison._direction_dict = {
            "accuracy": "maximize",
            "precision": "maximize",
            "recall": "maximize",
            "f1": "maximize",
            "roc_auc": "maximize",
            "pr_auc": "maximize",
        }
    elif task_type == "regression":
        PostHocComparison._metrics_names = ["mse", "mae", "r2", "ktau", "spearmanr"]
        PostHocComparison._direction_dict = {
            "mae": "minimize",
            "mse": "minimize",
            "r2": "maximize",
            "ktau": "maximize",
            "spearmanr": "maximize",
        }


def compare_models(
    runs_root: Path,
    out_root: Path,
    seed: int | None,
    label_types: list[str],
    dry_run: bool,
    task_type: str,
) -> None:
    runs_root = runs_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    grouped = _iter_run_dirs(runs_root, seed)
    for dataset, run_dirs in grouped.items():
        if not run_dirs:
            continue
        dataset_out = out_root / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        if task_type == "classification":
            _configure_compare_metrics(task_type)
            if dry_run:
                print(
                    "PostHocComparison.compare("
                    f"model_dirs={len(run_dirs)}, "
                    f"label_types={label_types}, "
                    f"output_dir={dataset_out}, report=True)"
                )
                continue
            comp = PostHocComparison()
            comp.compare(
                model_dirs=[str(p) for p in run_dirs],
                label_types=label_types,
                output_dir=str(dataset_out),
                report=True,
            )
        else:
            cmd = [
                "openadmet",
                "compare",
                *(_flatten_args("--model-dirs", [str(p) for p in run_dirs])),
                *(_flatten_args("--label-types", label_types)),
                "--output-dir",
                str(dataset_out),
                "--report",
                "True",
            ]

            if dry_run:
                print(" ".join(cmd))
                continue
            subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1, help="Only compare runs for this seed (default: 1).")
    ap.add_argument(
        "--label-types",
        nargs="+",
        default=["biotarget", "model", "feat"],
        help="Label types for compare (default: biotarget model feat).",
    )
    ap.add_argument(
        "--task-type",
        choices=["regression", "classification"],
        default="regression",
        help="Metric family for openadmet compare logic.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    compare_models(
        runs_root=args.runs_root,
        out_root=args.out_root,
        seed=args.seed,
        label_types=list(args.label_types),
        dry_run=args.dry_run,
        task_type=args.task_type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
