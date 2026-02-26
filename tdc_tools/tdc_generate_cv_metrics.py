from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from openadmet.models.anvil.specification import DataSpec, ProcedureSpec
from openadmet.models.architecture.model_base import get_mod_class
from openadmet.models.eval import cross_validation as cv_module
from openadmet.models.eval.cross_validation import (
    PytorchLightningRepeatedKFoldCrossValidation,
    SKLearnRepeatedKFoldCrossValidation,
)
from openadmet.models.inference.inference import load_anvil_model_and_metadata
from openadmet.models.trainer.lightning import LightningTrainer


SKLEARN_CV_TYPES = {"SKLearnRepeatedKFoldCrossValidation"}
LIGHTNING_CV_TYPES = {"PytorchLightningRepeatedKFoldCrossValidation"}


class _SklearnModelAdapter:
    """Adapter exposing both .estimator and sklearn prediction methods."""

    def __init__(self, estimator):
        self.estimator = estimator

    def __getattr__(self, name: str):
        return getattr(self.estimator, name)


def _ensure_2d(arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _featurize_simple(feat, X):
    out = feat.featurize(X)
    if isinstance(out, tuple):
        return out[0]
    return out


def _load_cv_eval(recipe: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    evals = recipe.get("report", {}).get("eval", [])
    for ev in evals:
        ev_type = str(ev.get("type", ""))
        if "CrossValidation" in ev_type:
            return ev_type, dict(ev.get("params") or {})
    return None, {}


def _detect_task_type(recipe: dict[str, Any]) -> str:
    evals = recipe.get("report", {}).get("eval", [])
    eval_types = [str(ev.get("type", "")) for ev in evals]
    if any("ClassificationMetrics" in ev_type for ev_type in eval_types):
        return "classification"
    model_type = str(recipe.get("procedure", {}).get("model", {}).get("type", ""))
    if "Classifier" in model_type:
        return "classification"
    return "regression"


def _flatten_binary(y: Any) -> np.ndarray:
    arr = np.asarray(y).reshape(-1)
    if arr.dtype.kind in {"b", "i", "u"}:
        return arr.astype(int)
    return (arr >= 0.5).astype(int)


def _flatten_scores(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[1] > 1:
            arr = arr[:, -1]
        else:
            arr = arr[:, 0]
    return arr.reshape(-1)


def _estimator_scores(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return _flatten_scores(estimator.predict_proba(X))
    if hasattr(estimator, "decision_function"):
        return _flatten_scores(estimator.decision_function(X))
    return _flatten_scores(estimator.predict(X))


def _sk_accuracy(estimator, X, y_true) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(estimator.predict(X))
    return float(accuracy_score(y_t, y_p))


def _sk_precision(estimator, X, y_true) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(estimator.predict(X))
    return float(precision_score(y_t, y_p, zero_division=0))


def _sk_recall(estimator, X, y_true) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(estimator.predict(X))
    return float(recall_score(y_t, y_p, zero_division=0))


def _sk_f1(estimator, X, y_true) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(estimator.predict(X))
    return float(f1_score(y_t, y_p, zero_division=0))


def _sk_roc_auc(estimator, X, y_true) -> float:
    y_t = _flatten_binary(y_true)
    y_s = _estimator_scores(estimator, X)
    try:
        return float(roc_auc_score(y_t, y_s))
    except ValueError:
        return float("nan")


def _sk_pr_auc(estimator, X, y_true) -> float:
    y_t = _flatten_binary(y_true)
    y_s = _estimator_scores(estimator, X)
    try:
        return float(average_precision_score(y_t, y_s))
    except ValueError:
        return float("nan")


def _pl_accuracy(y_true, y_pred) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(y_pred)
    return float(accuracy_score(y_t, y_p))


def _pl_precision(y_true, y_pred) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(y_pred)
    return float(precision_score(y_t, y_p, zero_division=0))


def _pl_recall(y_true, y_pred) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(y_pred)
    return float(recall_score(y_t, y_p, zero_division=0))


def _pl_f1(y_true, y_pred) -> float:
    y_t = _flatten_binary(y_true)
    y_p = _flatten_binary(y_pred)
    return float(f1_score(y_t, y_p, zero_division=0))


def _pl_roc_auc(y_true, y_pred) -> float:
    y_t = _flatten_binary(y_true)
    y_s = _flatten_scores(y_pred)
    try:
        return float(roc_auc_score(y_t, y_s))
    except ValueError:
        return float("nan")


def _pl_pr_auc(y_true, y_pred) -> float:
    y_t = _flatten_binary(y_true)
    y_s = _flatten_scores(y_pred)
    try:
        return float(average_precision_score(y_t, y_s))
    except ValueError:
        return float("nan")


def _configure_cv_metrics(task_type: str) -> None:
    if task_type == "classification":
        cv_module.CrossValidationBase._metrics = {
            "accuracy": (_sk_accuracy, False, "Accuracy"),
            "precision": (_sk_precision, False, "Precision"),
            "recall": (_sk_recall, False, "Recall"),
            "f1": (_sk_f1, False, "F1"),
            "roc_auc": (_sk_roc_auc, False, "ROC-AUC"),
            "pr_auc": (_sk_pr_auc, False, "PR-AUC"),
        }
        cv_module.PytorchLightningRepeatedKFoldCrossValidation._metrics = {
            "accuracy": (_pl_accuracy, False, "Accuracy"),
            "precision": (_pl_precision, False, "Precision"),
            "recall": (_pl_recall, False, "Recall"),
            "f1": (_pl_f1, False, "F1"),
            "roc_auc": (_pl_roc_auc, False, "ROC-AUC"),
            "pr_auc": (_pl_pr_auc, False, "PR-AUC"),
        }
    else:
        cv_module.CrossValidationBase._metrics = {
            "mse": (cv_module.make_scorer(cv_module.mean_squared_error), False, "MSE"),
            "mae": (cv_module.make_scorer(cv_module.mean_absolute_error), False, "MAE"),
            "r2": (cv_module.make_scorer(cv_module.r2_score), False, "$R^2$"),
            "ktau": (cv_module.make_scorer(cv_module.wrap_ktau), True, "Kendall's $\\tau$"),
            "spearmanr": (
                cv_module.make_scorer(cv_module.wrap_spearmanr),
                True,
                "Spearman's $\\rho$",
            ),
        }
        cv_module.PytorchLightningRepeatedKFoldCrossValidation._metrics = {
            "mse": (cv_module.mean_squared_error, False, "MSE"),
            "mae": (cv_module.mean_absolute_error, False, "MAE"),
            "r2": (cv_module.r2_score, False, "$R^2$"),
            "ktau": (cv_module.wrap_ktau, True, "Kendall's $\\tau$"),
            "spearmanr": (cv_module.wrap_spearmanr, True, "Spearman's $\\rho$"),
        }


def _load_recipe(run_dir: Path) -> dict[str, Any]:
    recipe_path = run_dir / "anvil_recipe.yaml"
    return yaml.safe_load(recipe_path.read_text())


def _load_specs(run_dir: Path) -> tuple[ProcedureSpec, DataSpec]:
    proc = ProcedureSpec.from_yaml(run_dir / "recipe_components" / "procedure.yaml")
    data = DataSpec.from_yaml(run_dir / "recipe_components" / "data.yaml")
    return proc, data


def _run_sklearn_cv(run_dir: Path, recipe: dict[str, Any], params: dict[str, Any]) -> None:
    try:
        loaded_model, feat, _, data_spec = load_anvil_model_and_metadata(run_dir)
    except Exception:
        # Some legacy DummyClassifier runs have model params incompatible with
        # current pydantic validation. Fall back to the serialized sklearn model.
        model_type = str(recipe.get("procedure", {}).get("model", {}).get("type", ""))
        model_pkl = run_dir / "model.pkl"
        if "DummyClassifierModel" not in model_type or not model_pkl.exists():
            raise
        proc, data_spec = _load_specs(run_dir)
        feat = proc.feat.to_class()
        try:
            loaded_model = _SklearnModelAdapter(joblib.load(model_pkl))
        except Exception:
            with model_pkl.open("rb") as f:
                loaded_model = _SklearnModelAdapter(pickle.load(f))

    X_train, X_val, X_test, y_train, y_val, y_test, X_all, y_all = data_spec.read()

    X_train_feat = _featurize_simple(feat, X_train)
    X_val_feat = _featurize_simple(feat, X_val) if X_val is not None else None
    X_test_feat = _featurize_simple(feat, X_test) if X_test is not None else None
    X_all_feat = _featurize_simple(feat, X_all)

    proc, _ = _load_specs(run_dir)
    transform = proc.transform.to_class() if proc.transform else None
    if transform is not None:
        transform.fit(X_train_feat)
        X_train_feat = transform.transform(X_train_feat)
        if X_val_feat is not None:
            X_val_feat = transform.transform(X_val_feat)
        if X_test_feat is not None:
            X_test_feat = transform.transform(X_test_feat)
        X_all_feat = transform.transform(X_all_feat)

    if X_test_feat is None:
        raise RuntimeError(f"{run_dir} has no test split; CV needs test predictions")

    if hasattr(loaded_model, "predict_proba"):
        y_pred = loaded_model.predict_proba(X_test_feat)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            y_pred = y_pred[:, -1]
    else:
        y_pred = loaded_model.predict(X_test_feat)

    y_pred = _ensure_2d(y_pred)
    y_true = _ensure_2d(y_test.to_numpy())
    y_train_arr = _ensure_2d(y_train.to_numpy())
    y_all_arr = _ensure_2d(y_all.to_numpy())

    tag = str(recipe.get("metadata", {}).get("tag", run_dir.name))
    target_cols = recipe.get("data", {}).get("target_cols", [])
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    evaluator = SKLearnRepeatedKFoldCrossValidation(**params)
    evaluator.evaluate(
        model=loaded_model,
        X_train=X_train_feat,
        y_train=y_train_arr,
        y_true=y_true,
        y_pred=y_pred,
        X_all=X_all_feat,
        y_all=y_all_arr,
        groups=None,
        tag=tag,
        target_labels=target_cols or None,
    )
    evaluator.report(write=True, output_dir=run_dir)


def _run_lightning_cv(
    run_dir: Path,
    recipe: dict[str, Any],
    params: dict[str, Any],
    lightning_accelerator: str,
    lightning_devices: int,
) -> None:
    proc, data_spec = _load_specs(run_dir)
    feat = proc.feat.to_class()

    X_train, X_val, X_test, y_train, y_val, y_test, X_all, y_all = data_spec.read()

    train_dl, _, train_scaler, _ = feat.featurize(X_train, y_train)
    test_dl, _, _, _ = feat.featurize(X_test, y_test)

    model_type = str(recipe.get("procedure", {}).get("model", {}).get("type", ""))
    model_cls = get_mod_class(model_type)
    model = model_cls.deserialize(
        param_path=run_dir / "model.json",
        serial_path=run_dir / "model.pth",
        scaler=train_scaler,
    )

    trainer = proc.train.to_class() if proc.train else LightningTrainer()
    if trainer.output_dir is None:
        trainer.output_dir = run_dir
    trainer.accelerator = lightning_accelerator
    trainer.devices = lightning_devices

    y_pred = model.predict(
        test_dl,
        accelerator=lightning_accelerator,
        devices=lightning_devices,
    )
    y_pred = _ensure_2d(y_pred)
    y_true = _ensure_2d(y_test.to_numpy())

    tag = str(recipe.get("metadata", {}).get("tag", run_dir.name))
    target_cols = recipe.get("data", {}).get("target_cols", [])
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    evaluator = PytorchLightningRepeatedKFoldCrossValidation(**params)
    evaluator.evaluate(
        model=model,
        X_train=X_train,
        y_train=y_train,
        y_true=y_true,
        y_pred=y_pred,
        X_all=X_all,
        y_all=y_all,
        groups=None,
        featurizer=feat,
        trainer=trainer,
        tag=tag,
        target_labels=target_cols or None,
    )
    evaluator.report(write=True, output_dir=run_dir)


def generate_cv(
    runs_root: Path,
    dataset: str,
    seed: int,
    overwrite: bool,
    exclude_models: set[str],
    task_type: str,
    lightning_accelerator: str,
    lightning_devices: int,
    require_classification_metrics: bool,
) -> None:
    runs_root = runs_root.expanduser().resolve()

    for tag_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        parts = tag_dir.name.split("__")
        if len(parts) < 6 or parts[0] != "tdc":
            continue
        if parts[1] != dataset or parts[2] != f"seed{seed}":
            continue
        model_type = parts[3]
        if model_type in exclude_models:
            continue

        for run_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()]):
            if require_classification_metrics and not (
                run_dir / "classification_metrics.json"
            ).exists():
                continue
            cv_path = run_dir / "cross_validation_metrics.json"
            if cv_path.exists() and not overwrite:
                continue

            recipe = _load_recipe(run_dir)
            this_task_type = task_type
            if this_task_type == "auto":
                this_task_type = _detect_task_type(recipe)
            _configure_cv_metrics(this_task_type)
            eval_type, params = _load_cv_eval(recipe)
            if not eval_type:
                continue

            if eval_type in SKLEARN_CV_TYPES:
                _run_sklearn_cv(run_dir, recipe, params)
            elif eval_type in LIGHTNING_CV_TYPES:
                _run_lightning_cv(
                    run_dir=run_dir,
                    recipe=recipe,
                    params=params,
                    lightning_accelerator=lightning_accelerator,
                    lightning_devices=lightning_devices,
                )
            else:
                raise ValueError(f"Unknown CV evaluator type: {eval_type}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=Path("tdc_runs"), type=Path)
    ap.add_argument("--dataset", default="ames")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--exclude-model",
        action="append",
        default=["TabPFNClassifierModel", "TabPFNPostHocClassifierModel"],
        help="Model type to skip (repeatable).",
    )
    ap.add_argument(
        "--task-type",
        choices=["auto", "regression", "classification"],
        default="auto",
        help="CV metric family to use. Default: auto (detect from recipe).",
    )
    ap.add_argument(
        "--lightning-accelerator",
        choices=["cpu", "gpu", "auto"],
        default="cpu",
        help="Accelerator for Lightning-based CV models (e.g., ChemProp).",
    )
    ap.add_argument(
        "--lightning-devices",
        type=int,
        default=1,
        help="Number of devices for Lightning-based CV models.",
    )
    ap.add_argument(
        "--require-classification-metrics",
        action="store_true",
        help="Only process run directories that already have classification_metrics.json.",
    )
    args = ap.parse_args()

    generate_cv(
        runs_root=args.runs_root,
        dataset=args.dataset,
        seed=args.seed,
        overwrite=args.overwrite,
        exclude_models=set(args.exclude_model or []),
        task_type=args.task_type,
        lightning_accelerator=args.lightning_accelerator,
        lightning_devices=args.lightning_devices,
        require_classification_metrics=args.require_classification_metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
