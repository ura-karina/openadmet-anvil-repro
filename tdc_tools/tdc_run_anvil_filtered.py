from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXCLUDE_MODELS = {
    "ChemPropModel",
    "TabPFNClassifierModel",
    "TabPFNPostHocClassifierModel",
}


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_recipes(recipes_root: Path) -> list[Path]:
    return sorted([p for p in recipes_root.rglob("recipe_*.yaml") if p.is_file()])


def _load_recipe(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _recipe_model_type(recipe: dict[str, Any]) -> str:
    return str(recipe.get("procedure", {}).get("model", {}).get("type", ""))


def _recipe_tag(recipe: dict[str, Any]) -> str:
    return str(recipe.get("metadata", {}).get("tag", ""))


def _recipe_name(recipe: dict[str, Any]) -> str:
    return str(recipe.get("metadata", {}).get("name", ""))


def _is_completed(run_dir: Path) -> bool:
    if not (run_dir / "anvil_recipe.yaml").exists():
        return False
    has_model = (run_dir / "model.json").exists() and (
        (run_dir / "model.pkl").exists() or (run_dir / "model.ckpt").exists()
    )
    has_metrics = any((run_dir / fn).exists() for fn in ("regression_metrics.json", "classification_metrics.json"))
    return has_model and has_metrics


def _should_exclude(recipe: dict[str, Any], exclude_models: set[str]) -> bool:
    model_type = _recipe_model_type(recipe)
    if model_type in exclude_models:
        return True
    tag = _recipe_tag(recipe)
    name = _recipe_name(recipe)
    return any(token in tag or token in name for token in exclude_models)


def run_filtered(recipes_root: Path, runs_root: Path, log_path: Path, exclude_models: set[str]) -> None:
    recipes = _iter_recipes(recipes_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for recipe_path in recipes:
        recipe = _load_recipe(recipe_path)
        if _should_exclude(recipe, exclude_models):
            continue

        recipe_hash = _sha1_file(recipe_path)[:12]
        tag = _recipe_tag(recipe)
        run_dir = runs_root / tag / recipe_hash

        if run_dir.exists() and _is_completed(run_dir):
            continue

        record: dict[str, Any] = {
            "recipe_path": str(recipe_path),
            "tag": tag,
            "recipe_hash": recipe_hash,
            "run_dir": str(run_dir),
            "t_start": time.time(),
            "status": "running",
        }
        log_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")

        run_dir.parent.mkdir(parents=True, exist_ok=True)

        # `openadmet anvil` will suffix the output dir if it exists; so ensure it doesn't.
        if run_dir.exists():
            record["status"] = "skipped_exists"
            record["t_end"] = time.time()
            log_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
            continue

        cmd = [
            "openadmet",
            "anvil",
            "--recipe-path",
            str(recipe_path),
            "--output-dir",
            str(run_dir),
        ]

        try:
            subprocess.run(cmd, check=True)
            record["status"] = "ok"
        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["returncode"] = exc.returncode
        record["t_end"] = time.time()
        log_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes-root", default=Path("tdc_recipes/ames"), type=Path)
    ap.add_argument("--runs-root", default=Path("tdc_runs"), type=Path)
    ap.add_argument("--log-path", default=Path("tdc_runs/logs/runs.jsonl"), type=Path)
    ap.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Model type substring to exclude (repeatable).",
    )
    args = ap.parse_args()

    exclude_models = set(args.exclude_model) if args.exclude_model else set(DEFAULT_EXCLUDE_MODELS)
    run_filtered(
        recipes_root=args.recipes_root,
        runs_root=args.runs_root,
        log_path=args.log_path,
        exclude_models=exclude_models,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
