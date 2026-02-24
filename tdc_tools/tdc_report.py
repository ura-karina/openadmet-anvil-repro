from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _best_anvil(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in df.groupby("dataset"):
        metric = sub["metric_primary"].iloc[0]
        higher_is_better = bool(sub["metric_primary_higher_is_better"].iloc[0])
        score_col = f"{metric}_mean"
        if score_col not in sub.columns:
            continue
        best = sub.sort_values(score_col, ascending=not higher_is_better).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows)


def build_report(leaderboard_csv: Path, baselines_csv: Path | None, out_md: Path) -> None:
    leaderboard_csv = leaderboard_csv.expanduser().resolve()
    out_md = out_md.expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(leaderboard_csv)
    best = _best_anvil(df)

    if baselines_csv and baselines_csv.exists():
        baselines = pd.read_csv(baselines_csv)
    else:
        baselines = pd.DataFrame(columns=["dataset", "metric", "value", "model", "source"])

    merged = best.merge(baselines, on="dataset", how="left", suffixes=("", "_baseline"))

    out_cols = [
        "dataset",
        "task_kind",
        "model_type",
        "feat_type",
        "hp_id",
        "metric_primary",
        "metric_primary_higher_is_better",
        "metric_primary_mean",
        "metric_primary_std",
        "model",
        "metric",
        "value",
        "source",
    ]
    for col in out_cols:
        if col not in merged.columns:
            merged[col] = ""

    table = merged[out_cols].copy()
    table = table.rename(
        columns={
            "model": "baseline_model",
            "metric": "baseline_metric",
            "value": "baseline_value",
            "source": "baseline_source",
        }
    )

    md = table.to_markdown(index=False)
    out_md.write_text(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", required=True, type=Path)
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--out-md", required=True, type=Path)
    args = ap.parse_args()

    build_report(leaderboard_csv=args.leaderboard, baselines_csv=args.baselines, out_md=args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
