# OpenADMET Anvil Repro Tools (Kolmogorov)

This folder contains helper scripts intended to be copied to the Kolmogorov server and executed inside the `oadmet2` micromamba environment.

Typical flow on server:

1) Quick sanity check that PyTDC ADMET_Group APIs are available:

```bash
~/bin/micromamba run -n oadmet2 python tdc_check.py
```

2) Export TDC ADMET_Group splits to parquet (train/val per-seed + fixed test):

```bash
~/bin/micromamba run -n oadmet2 python tdc_export_admet_group.py \
  --out-root ~/openadmet-anvil-repro/tdc_data \
  --seeds 1 2 3 4 5
```

3) Generate Anvil recipes (matrix of dataset × seed × model × feature-set):

```bash
~/bin/micromamba run -n oadmet2 python tdc_generate_recipes.py \
  --data-root ~/openadmet-anvil-repro/tdc_data \
  --out-root ~/openadmet-anvil-repro/tdc_recipes \
  --matrix-path ~/openadmet-anvil-repro/tdc_recipes/matrix.yaml
```

4) Run all recipes (resume by skipping existing output dirs):

```bash
~/bin/micromamba run -n oadmet2 python tdc_run_anvil_matrix.py \
  --recipes-root ~/openadmet-anvil-repro/tdc_recipes \
  --runs-root ~/openadmet-anvil-repro/tdc_runs
```

5) Evaluate using test-set predictions and write a leaderboard:

```bash
~/bin/micromamba run -n oadmet2 python tdc_evaluate.py \
  --data-root ~/openadmet-anvil-repro/tdc_data \
  --runs-root ~/openadmet-anvil-repro/tdc_runs \
  --out-csv ~/openadmet-anvil-repro/tdc_leaderboards/leaderboard.csv
```

6) Compare models using CV metrics (one dataset per report):

```bash
~/bin/micromamba run -n oadmet2 python tdc_compare.py \
  --runs-root ~/openadmet-anvil-repro/tdc_runs \
  --out-root ~/openadmet-anvil-repro/tdc_leaderboards/compare
```

7) Fetch baseline results (best-effort) and build a summary report:

```bash
~/bin/micromamba run -n oadmet2 python tdc_fetch_baselines.py \
  --data-root ~/openadmet-anvil-repro/tdc_data \
  --out-csv ~/openadmet-anvil-repro/tdc_leaderboards/tdc_baselines.csv

~/bin/micromamba run -n oadmet2 python tdc_report.py \
  --leaderboard ~/openadmet-anvil-repro/tdc_leaderboards/leaderboard.csv \
  --baselines ~/openadmet-anvil-repro/tdc_leaderboards/tdc_baselines.csv \
  --out-md ~/openadmet-anvil-repro/tdc_leaderboards/summary.md
```

8) (Optional) Generate ensemble recipes for top regression models:

```bash
~/bin/micromamba run -n oadmet2 python tdc_generate_ensembles.py \
  --runs-root ~/openadmet-anvil-repro/tdc_runs \
  --leaderboard ~/openadmet-anvil-repro/tdc_leaderboards/leaderboard.csv \
  --out-root ~/openadmet-anvil-repro/tdc_recipes_ensembles

~/bin/micromamba run -n oadmet2 python tdc_run_anvil_matrix.py \
  --recipes-root ~/openadmet-anvil-repro/tdc_recipes_ensembles \
  --runs-root ~/openadmet-anvil-repro/tdc_runs
```
