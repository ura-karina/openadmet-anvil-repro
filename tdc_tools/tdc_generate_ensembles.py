from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _ensemble_type(task_kind: str) -> str:
    return "CommitteeClassifier" if task_kind == "classification" else "CommitteeRegressor"


def _find_run_dir(runs_root: Path, tag: str) -> Path | None:
    tag_dir = runs_root / tag
    if not tag_dir.exists():
        return None
    for run_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()]):
        if (run_dir / "anvil_recipe.yaml").exists():
            return run_dir
    return None


def generate_ensembles(
    runs_root: Path,
    leaderboard_csv: Path,
    out_root: Path,
    top_n: int,
    seed: int,
    task_kind: str,
    n_models: int,
) -> int:
    runs_root = runs_root.expanduser().resolve()
    leaderboard_csv = leaderboard_csv.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(leaderboard_csv)
    df = df[df["task_kind"] == task_kind]
    if df.empty:
        return 0

    count = 0
    for dataset, sub in df.groupby("dataset"):
        metric = sub["metric_primary"].iloc[0]
        higher_is_better = bool(sub["metric_primary_higher_is_better"].iloc[0])
        score_col = f"{metric}_mean"
        if score_col not in sub.columns:
            continue
        sub = sub.sort_values(score_col, ascending=not higher_is_better)
        for _, row in sub.head(top_n).iterrows():
            model_type = row["model_type"]
            feat_type = row["feat_type"]
            hp_id = row["hp_id"]
            tag = f"tdc__{dataset}__seed{seed}__{model_type}__{feat_type}__{hp_id}"

            run_dir = _find_run_dir(runs_root, tag)
            if run_dir is None:
                continue

            recipe_path = run_dir / "anvil_recipe.yaml"
            recipe = yaml.safe_load(recipe_path.read_text())

            procedure: dict[str, Any] = recipe.get("procedure", {})
            if "ensemble" in procedure:
                continue

            procedure["ensemble"] = {"n_models": n_models, "type": _ensemble_type(task_kind)}
            recipe["procedure"] = procedure

            recipe["metadata"]["name"] = f"{dataset}__{model_type}__{feat_type}__{hp_id}__ensemble"
            recipe["metadata"]["tag"] = f"tdc_ens__{dataset}__seed{seed}__{model_type}__{feat_type}__{hp_id}"

            recipe_text = yaml.safe_dump(recipe, sort_keys=True)
            recipe_hash = _sha1_text(recipe_text)[:12]
            rel = Path(dataset) / f"seed_{seed}" / model_type / feat_type / hp_id / "ensemble"
            out_path = out_root / rel / f"recipe_{recipe_hash}.yaml"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(recipe_text)
            count += 1

    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--leaderboard", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--task-kind", choices=["regression", "classification"], default="regression")
    ap.add_argument("--n-models", type=int, default=3)
    args = ap.parse_args()

    count = generate_ensembles(
        runs_root=args.runs_root,
        leaderboard_csv=args.leaderboard,
        out_root=args.out_root,
        top_n=args.top_n,
        seed=args.seed,
        task_kind=args.task_kind,
        n_models=args.n_models,
    )
    print(f"Generated {count} ensemble recipes in {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
