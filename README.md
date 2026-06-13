# Sofia ML Regression 2026 Spring

Sofia University Machine Learning class — Kaggle regression homework.

## Problem

15 anonymized features (x0–x14) → one real-valued target, scored by R². The target has
extreme outliers (range −41,008 to +69,628, skewness 13.6) and the signal is weak
(max |feature–target correlation| ≈ 0.25).

## Approach

Rather than reaching for a more complex model, I cleaned the **training** target with
winsorization and kept a simple linear model:

```
median impute → standardize → winsorize target [0.5%, 99.5%] → Ridge (alpha = 1)
```

The reproducing script is `src/phase7b_winsor_all.py` (version v7b).

## Results

| Split | Score | Baseline | Outcome |
|-------|-------|----------|---------|
| Public LB  | 0.04843 | 0.04599 | Beat baseline (was #1 on public for a time) |
| Private LB | 0.05008 | 0.05258 | 6th; just under the private baseline |

Note on the private split: the leaderboard reshuffled heavily — two teams with negative
public scores finished 3rd and 4th, and the public #1 dropped to 5th. My entry was the
only one near the top whose rank did not move on either split (public 6th → private 6th),
which I take as evidence the model generalized rather than fit one lucky split. With an R²
ceiling around 0.05–0.08, most of the variance is irreducible noise, so a stable model
matters more than a high score on a single split.

## Repository structure

```
sofia-ml-regression/
├── README.md            This file
├── experiments.md       One-line log of every experiment
├── data/                train / test / sample_submission CSV
├── src/                 All experiment scripts
│   ├── eda.py
│   ├── baseline_lsm.py
│   ├── phase2_ridge_polynomial.py
│   ├── phase3_target_transform.py
│   ├── phase4_sweep.py
│   ├── phase5_push.py
│   ├── phase6_ensemble.py
│   ├── phase7_beat_baseline.py
│   ├── phase7b_winsor_all.py     Final submission (v7b)
│   ├── phase7c_winsor_poly.py
│   └── phase7d_lighter_winsor.py
├── submissions/         All Kaggle submission CSVs
└── docs/
    ├── Baek_Seunghan_Sofia_ML_Regression.pdf   Presentation deck
    ├── slide-deck.html                          Deck source
    ├── eda_plots/                               EDA figures
    └── phase*_results.csv                       Experiment result tables
```

## Reproducing the final score

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # numpy, pandas, scikit-learn
python src/phase7b_winsor_all.py
```

## Competition rules

- Metric: R²
- Submission format: `Id,target` CSV (2500 rows)
- Daily limit: 5 submissions
- Naming convention: `Baek_Seunghan` (Last_First)
- Scoring: beating the baseline (0.04599 public) earns full credit; top-3 earns a bonus
- Code reproducibility and a short presentation are both required
