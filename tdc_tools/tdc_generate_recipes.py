from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RecipeSpec:
    dataset_name: str
    seed: int
    task_kind: str  # regression|classification|unknown
    model_type: str
    feat_type: str
    hp_id: str
    recipe_path: Path


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_meta(dataset_dir: Path) -> dict[str, Any]:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _task_kind_from_meta(meta: dict[str, Any]) -> str:
    raw = str(meta.get("task_type") or meta.get("task") or "").lower()
    if "regress" in raw:
        return "regression"
    if "class" in raw:
        return "classification"
    return "unknown"


def _base_recipe(
    dataset_dir: Path,
    dataset_name: str,
    seed: int,
    model_type: str,
    feat_type: str,
    feat_params: dict[str, Any],
    trainer_type: str,
    trainer_params: dict[str, Any],
    task_kind: str,
    model_params: dict[str, Any],
) -> dict[str, Any]:
    seed_dir = dataset_dir / f"seed_{seed}"

    data = {
        "type": "intake",
        "train_resource": str(seed_dir / "train.parquet"),
        "val_resource": str(seed_dir / "valid.parquet"),
        "test_resource": str(dataset_dir / "test.parquet"),
        "input_col": "OPENADMET_SMILES",
        "target_cols": ["y"],
        "dropna": True,
    }

    metadata = {
        "authors": "openadmet-anvil-repro",
        "email": "n/a",
        "biotargets": [dataset_name],
        "build_number": 0,
        "description": f"TDC ADMET_Group {dataset_name} seed {seed}",
        "driver": "sklearn",
        "name": f"{dataset_name}__{model_type}__{feat_type}",
        # tag controls prediction column names; keep deterministic.
        "tag": f"tdc__{dataset_name}__seed{seed}__{model_type}__{feat_type}",
        "tags": ["tdc", "admet_group"],
        "version": "v1",
    }

    # No-op split because we are using train/val/test resources.
    split = {
        "type": "ShuffleSplitter",
        "params": {"random_state": 42, "train_size": 1.0, "val_size": 0.0, "test_size": 0.0},
    }

    procedure: dict[str, Any] = {
        "feat": {"type": feat_type, "params": feat_params},
        "model": {"type": model_type, "params": model_params},
        "split": split,
        "train": {"type": trainer_type, "params": trainer_params},
    }

    # Descriptor featurizers frequently produce NaNs; impute for stability.
    if feat_type == "DescriptorFeaturizer":
        procedure["transform"] = {"type": "ImputeTransform", "params": {"strategy": "mean"}}

    report_eval: list[dict[str, Any]] = []
    if task_kind == "classification":
        report_eval.append({"type": "ClassificationMetrics", "params": {}})
    else:
        report_eval.append({"type": "RegressionMetrics", "params": {}})

    cv_type = "PytorchLightningRepeatedKFoldCrossValidation" if model_type == "ChemPropModel" else "SKLearnRepeatedKFoldCrossValidation"
    report_eval.append(
        {
            "type": cv_type,
            "params": {
                "n_splits": 5,
                "n_repeats": 5,
                "random_state": 42,
            },
        }
    )

    recipe: dict[str, Any] = {"data": data, "metadata": metadata, "procedure": procedure, "report": {"eval": report_eval}}
    return recipe


def _load_matrix(matrix_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(matrix_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid matrix config: {matrix_path}")
    return data


def _normalize_task_kinds(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _normalize_seeds(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[int] = []
        for item in raw:
            if item is None:
                continue
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid seed entry: {item!r}")
        return out
    try:
        return [int(raw)]
    except (TypeError, ValueError):
        raise ValueError(f"Invalid seeds value: {raw!r}")


def _normalize_dataset_specs(raw: Any) -> tuple[list[str], dict[str, str]]:
    if raw is None:
        return [], {}
    if not isinstance(raw, list):
        raise ValueError("matrix.datasets must be a list of {name, task_kind} entries")
    names: list[str] = []
    task_map: dict[str, str] = {}
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"matrix.datasets[{idx}] must be a mapping with name/task_kind")
        name = entry.get("name") or entry.get("dataset") or entry.get("dataset_name")
        task_kind = entry.get("task_kind") or entry.get("task")
        if not name or not task_kind:
            raise ValueError(f"matrix.datasets[{idx}] must include name and task_kind")
        task_kind = str(task_kind).lower()
        if task_kind not in {"regression", "classification"}:
            raise ValueError(f"matrix.datasets[{idx}] has invalid task_kind={task_kind!r}")
        name = str(name)
        if name in task_map:
            raise ValueError(f"Duplicate dataset entry for {name}")
        names.append(name)
        task_map[name] = task_kind
    return names, task_map


def generate_recipes(data_root: Path, out_root: Path, seeds: list[int] | None, matrix_path: Path) -> list[RecipeSpec]:
    data_root = data_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    matrix = _load_matrix(matrix_path)
    matrix_seeds = _normalize_seeds(matrix.get("seeds"))
    if not seeds:
        seeds = matrix_seeds if matrix_seeds else [1, 2, 3, 4, 5]
    else:
        seeds = list(seeds)

    dataset_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    dataset_dir_map = {p.name: p for p in dataset_dirs}
    dataset_names, task_overrides = _normalize_dataset_specs(matrix.get("datasets"))
    if dataset_names:
        missing = sorted(set(dataset_names) - set(dataset_dir_map))
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(f"Missing dataset directories for: {missing_str}")
        dataset_dirs = [dataset_dir_map[name] for name in dataset_names]
    else:
        dataset_dirs = sorted(dataset_dirs, key=lambda p: p.name)

    feat_map: dict[str, dict[str, Any]] = matrix.get("featurizers", {})
    trainer_map: dict[str, dict[str, Any]] = matrix.get("trainers", {})
    models_list: list[dict[str, Any]] = matrix.get("models", [])

    specs: list[RecipeSpec] = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        task_kind = task_overrides.get(dataset_name)
        if task_kind is None:
            meta = _load_meta(dataset_dir)
            task_kind = _task_kind_from_meta(meta)
            if task_kind == "unknown":
                # Default to regression (some ADMET tasks are regression; evaluation script will handle).
                task_kind = "regression"

        task_models = []
        for model_cfg in models_list:
            task_kinds = _normalize_task_kinds(model_cfg.get("task_kind") or model_cfg.get("task_kinds"))
            if not task_kinds:
                continue
            if task_kind not in task_kinds:
                continue
            task_models.append(model_cfg)

        for seed in seeds:
            for model_cfg in task_models:
                model_type = str(model_cfg.get("name"))
                model_params = dict(model_cfg.get("model_params") or {})
                for feat_id in model_cfg.get("featurizers", []):
                    feat_cfg = feat_map.get(feat_id)
                    if not isinstance(feat_cfg, dict):
                        continue
                    feat_type = str(feat_cfg.get("type"))
                    feat_params = dict(feat_cfg.get("params") or {})

                    for trainer_id in model_cfg.get("trainers", []):
                        trainer_cfg = trainer_map.get(trainer_id)
                        if not isinstance(trainer_cfg, dict):
                            continue
                        trainer_type = str(trainer_cfg.get("type"))
                        trainer_params = dict(trainer_cfg.get("params") or {})

                        recipe = _base_recipe(
                            dataset_dir=dataset_dir,
                            dataset_name=dataset_name,
                            seed=seed,
                            model_type=model_type,
                            feat_type=feat_type,
                            feat_params=feat_params,
                            trainer_type=trainer_type,
                            trainer_params=trainer_params,
                            task_kind=task_kind,
                            model_params=model_params,
                        )

                        # Update tag/name to include featurizer + hp id for uniqueness.
                        recipe["metadata"]["name"] = f"{dataset_name}__{model_type}__{feat_id}__{trainer_id}"
                        recipe["metadata"]["tag"] = (
                            f"tdc__{dataset_name}__seed{seed}__{model_type}__{feat_id}__{trainer_id}"
                        )

                        # Apply imputation when descriptor featurizers are present (including concatenation).
                        if feat_type == "DescriptorFeaturizer":
                            recipe["procedure"]["transform"] = {"type": "ImputeTransform", "params": {"strategy": "mean"}}
                        if feat_type == "FeatureConcatenator":
                            for sub_feat in feat_params.get("featurizers", []):
                                if str(sub_feat.get("type")) == "DescriptorFeaturizer":
                                    recipe["procedure"]["transform"] = {
                                        "type": "ImputeTransform",
                                        "params": {"strategy": "mean"},
                                    }
                                    break

                        # Deterministic recipe file name.
                        recipe_text = yaml.safe_dump(recipe, sort_keys=True)
                        recipe_hash = _sha1_text(recipe_text)[:12]
                        rel = Path(dataset_name) / f"seed_{seed}" / model_type / feat_id / trainer_id
                        recipe_path = out_root / rel / f"recipe_{recipe_hash}.yaml"
                        recipe_path.parent.mkdir(parents=True, exist_ok=True)
                        recipe_path.write_text(recipe_text)

                        specs.append(
                            RecipeSpec(
                                dataset_name=dataset_name,
                                seed=seed,
                                task_kind=task_kind,
                                model_type=model_type,
                                feat_type=feat_id,
                                hp_id=trainer_id,
                                recipe_path=recipe_path,
                            )
                        )

    manifest = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "seeds": seeds,
        "matrix_path": str(matrix_path),
        "n_recipes": len(specs),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--matrix-path", default=Path("tdc_recipes/matrix.yaml"), type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()

    specs = generate_recipes(
        data_root=args.data_root,
        out_root=args.out_root,
        seeds=list(args.seeds) if args.seeds else None,
        matrix_path=args.matrix_path,
    )
    print(f"Generated {len(specs)} recipes in {Path(args.out_root).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
