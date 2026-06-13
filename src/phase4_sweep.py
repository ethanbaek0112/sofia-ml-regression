"""
Phase 4: Comprehensive Sweep — LASSO, ElasticNet, high-α Ridge, k-NN
====================================================================

Lessons learned so far
----------------------
- Phase 1 (plain LSM)              : CV R² = -0.457
- Phase 2 (Poly+Ridge, best α=1k)  : CV R² = -0.839  (worse: poly amplifies outliers)
- Phase 3 (signed-log target)      : CV R² = -25.95  (catastrophic: inverse blows up)

Target has 10 rows with |y| > 10,000 and 0.4% of rows that dominate R².
=> The right approach is probably aggressive shrinkage (high α) that keeps
   predictions near the median, sacrificing those extreme rows.

Models tested here (all from Lec 06 and Lec 07):
  - Ridge with extended α range (Lec 07)
  - LASSO with extended α range  (Lec 07)  ← provides feature selection
  - ElasticNet (Lec 07)                    ← combines L1+L2
  - k-NN Regression (Lec 06)               ← non-parametric baseline
All evaluated by 5-fold CV (Lec 02). No target transformation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"

SEED = 42
N_FOLDS = 5

FEATURE_COLS = [f"x{i}" for i in range(15)]
TARGET_COL = "target"
ID_COL = "Id"
SUBMISSION_NAME = "Baek_Seunghan"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def cv_score(pipe: Pipeline, X, y, cv: KFold) -> tuple[float, float]:
    s = cross_val_score(pipe, X, y, scoring="r2", cv=cv, n_jobs=-1)
    return float(s.mean()), float(s.std())


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
section("1. LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X_train = train[FEATURE_COLS].values
y_train = train[TARGET_COL].values
X_test = test[FEATURE_COLS].values
test_ids = test[ID_COL].values
print(f"train: {train.shape}, test: {test.shape}")

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results: list[dict] = []


# ---------------------------------------------------------------------------
# 2. Ridge with extended α range
# ---------------------------------------------------------------------------
section("2. RIDGE (L2) — α sweep")
for alpha in [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0]:
    pipe = build(Ridge(alpha=alpha, random_state=SEED))
    mean, std = cv_score(pipe, X_train, y_train, cv)
    print(f"  α={alpha:>10g}: CV R² = {mean:+.5f} ± {std:.5f}")
    results.append({"family": "Ridge", "config": f"α={alpha:g}",
                    "cv_mean": mean, "cv_std": std,
                    "model_factory": lambda a=alpha: Ridge(alpha=a, random_state=SEED)})


# ---------------------------------------------------------------------------
# 3. LASSO (L1) — α sweep (provides feature selection)
# ---------------------------------------------------------------------------
section("3. LASSO (L1) — α sweep")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    pipe = build(Lasso(alpha=alpha, random_state=SEED, max_iter=20000))
    mean, std = cv_score(pipe, X_train, y_train, cv)
    print(f"  α={alpha:>8g}: CV R² = {mean:+.5f} ± {std:.5f}")
    results.append({"family": "LASSO", "config": f"α={alpha:g}",
                    "cv_mean": mean, "cv_std": std,
                    "model_factory": lambda a=alpha: Lasso(alpha=a, random_state=SEED, max_iter=20000)})


# ---------------------------------------------------------------------------
# 4. ElasticNet — (α, l1_ratio) sweep
# ---------------------------------------------------------------------------
section("4. ELASTIC NET — (α, l1_ratio) sweep")
for alpha in [0.1, 1.0, 10.0, 100.0]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        pipe = build(ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                random_state=SEED, max_iter=20000))
        mean, std = cv_score(pipe, X_train, y_train, cv)
        print(f"  α={alpha:>6g}, l1_ratio={l1_ratio:>3}: "
              f"CV R² = {mean:+.5f} ± {std:.5f}")
        results.append({
            "family": "ElasticNet",
            "config": f"α={alpha:g}, l1_ratio={l1_ratio}",
            "cv_mean": mean, "cv_std": std,
            "model_factory": lambda a=alpha, r=l1_ratio: ElasticNet(
                alpha=a, l1_ratio=r, random_state=SEED, max_iter=20000),
        })


# ---------------------------------------------------------------------------
# 5. k-NN Regression (Lec 06) — k sweep
# ---------------------------------------------------------------------------
section("5. k-NN REGRESSION — k sweep")
for k in [3, 5, 10, 20, 50, 100, 200]:
    for weights in ["uniform", "distance"]:
        pipe = build(KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1))
        mean, std = cv_score(pipe, X_train, y_train, cv)
        print(f"  k={k:>3}, weights={weights:<8}: "
              f"CV R² = {mean:+.5f} ± {std:.5f}")
        results.append({
            "family": "k-NN",
            "config": f"k={k}, weights={weights}",
            "cv_mean": mean, "cv_std": std,
            "model_factory": lambda kk=k, ww=weights: KNeighborsRegressor(
                n_neighbors=kk, weights=ww, n_jobs=-1),
        })


# ---------------------------------------------------------------------------
# 6. Pick winner
# ---------------------------------------------------------------------------
section("6. RANKING")
results_df = pd.DataFrame(results).sort_values("cv_mean", ascending=False)
print(results_df.drop(columns=["model_factory"]).head(15).to_string(index=False))

winner = results_df.iloc[0]
print(f"\nWinner: {winner['family']} ({winner['config']})")
print(f"CV R² = {winner['cv_mean']:+.5f} ± {winner['cv_std']:.5f}")


# ---------------------------------------------------------------------------
# 7. Fit winner on full data + submission
# ---------------------------------------------------------------------------
section("7. FIT WINNER + SUBMISSION")
best_pipe = build(winner["model_factory"]())
best_pipe.fit(X_train, y_train)
train_pred = best_pipe.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
print(f"Train R² (in-sample): {train_r2:+.5f}")
print(f"CV R² mean         : {winner['cv_mean']:+.5f}")
print(f"Generalization gap : {train_r2 - winner['cv_mean']:+.5f}")

test_pred = best_pipe.predict(X_test)
print(f"\nTest pred stats: mean={test_pred.mean():.2f}, std={test_pred.std():.2f}, "
      f"min={test_pred.min():.2f}, max={test_pred.max():.2f}")

submission = pd.DataFrame({ID_COL: test_ids, "target": test_pred})
submission = submission.set_index(ID_COL).loc[sample[ID_COL]].reset_index()

family_short = winner["family"].lower().replace("-", "")
config_short = winner["config"].replace(", ", "_").replace("=", "").replace(" ", "")
version = f"v04_{family_short}_{config_short}"
out_path = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_{version}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(submission.head())


# ---------------------------------------------------------------------------
# 8. Save full results table for analysis
# ---------------------------------------------------------------------------
results_df.drop(columns=["model_factory"]).to_csv(
    ROOT / "docs" / "phase4_results.csv", index=False)
print(f"\nFull results: {ROOT / 'docs' / 'phase4_results.csv'}")

section("9. EXPERIMENTS.MD LINE")
print(
    f"| 04 | (today) | {winner['family']} ({winner['config']}) "
    f"| {winner['cv_mean']:+.5f} ± {winner['cv_std']:.5f} | TBD "
    f"| {out_path.name} | Comprehensive sweep, raw features, no target transform. |"
)
