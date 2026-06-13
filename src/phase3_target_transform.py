"""
Phase 3: Target Transformation + Ridge — diagnosing the outlier problem
======================================================================

Why
---
Phase 2 (Polynomial Ridge) actually DID WORSE than baseline because the target
has extreme outliers (range -41008 to +69628, kurtosis 708). When polynomial
features amplify input variance, the model overfits to these outliers in train
folds and collapses on validation folds.

Approach
--------
1. Diagnose: are extreme targets a pattern (predictable from features) or noise?
2. Apply a SIGNED LOG transformation:  y' = sign(y) * log1p(|y|)
   This is a monotonic, invertible transformation that compresses the tails
   while preserving order. Conceptually similar to MSLE in Lec 07 slide 23,
   but adapted to handle negative target values.
3. Train Ridge on transformed target, invert at prediction time.
4. Also sanity-check: median-only predictor (R² = 0 by definition on train).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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


def signed_log(y: np.ndarray) -> np.ndarray:
    """Symmetric log transform: preserves sign, compresses magnitude."""
    return np.sign(y) * np.log1p(np.abs(y))


def signed_exp(y: np.ndarray) -> np.ndarray:
    """Inverse of signed_log."""
    return np.sign(y) * (np.expm1(np.abs(y)))


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


# ---------------------------------------------------------------------------
# 2. Diagnose: outlier structure
# ---------------------------------------------------------------------------
section("2. TARGET OUTLIER DIAGNOSTIC")
percentiles = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
for p in percentiles:
    print(f"  p{p:>5}: {np.percentile(y_train, p):>14,.2f}")

# How many rows are 'extreme'?
abs_y = np.abs(y_train)
for thr in [100, 500, 1000, 5000, 10000]:
    n = (abs_y > thr).sum()
    print(f"  |y| > {thr:>5}: {n:>4} rows ({100*n/len(y_train):.2f}%)")


# ---------------------------------------------------------------------------
# 3. Sanity check: median-only predictor
# ---------------------------------------------------------------------------
section("3. SANITY CHECK: median-only predictor (R² on train)")
y_median_pred = np.full_like(y_train, np.median(y_train), dtype=float)
print(f"In-sample R² (median predictor): {r2_score(y_train, y_median_pred):.5f}")
print(f"In-sample R² (mean predictor)  : "
      f"{r2_score(y_train, np.full_like(y_train, y_train.mean(), dtype=float)):.5f}")
# Note: mean predictor gives R² = 0 by definition.


# ---------------------------------------------------------------------------
# 4. Effect of signed-log transformation
# ---------------------------------------------------------------------------
section("4. SIGNED-LOG TRANSFORM")
y_log = signed_log(y_train)
print(f"Original target  : mean={y_train.mean():.2f}, std={y_train.std():.2f}, "
      f"min={y_train.min():.2f}, max={y_train.max():.2f}, skew=...")
print(f"Signed-log target: mean={y_log.mean():.4f}, std={y_log.std():.4f}, "
      f"min={y_log.min():.4f}, max={y_log.max():.4f}")
print(f"Sanity check: signed_exp(signed_log(y))[:5] = "
      f"{signed_exp(signed_log(y_train[:5]))}")
print(f"             original           y[:5] = {y_train[:5]}")


# ---------------------------------------------------------------------------
# 5. Custom CV: train on log-space, evaluate R² in original space
# ---------------------------------------------------------------------------
section("5. CUSTOM CV: Ridge on log-space target, score on ORIGINAL space")


def cv_ridge_logspace(X, y_raw, degree: int, alpha: float, cv: KFold) -> dict:
    """Train Ridge on signed-log(y); inverse-transform predictions; score R² on y_raw."""
    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y_raw[tr_idx], y_raw[va_idx]
        y_tr_log = signed_log(y_tr)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=SEED)),
        ])
        pipe.fit(X_tr, y_tr_log)
        pred_log = pipe.predict(X_va)
        pred = signed_exp(pred_log)
        score = r2_score(y_va, pred)
        fold_scores.append(score)
    return {
        "scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
    }


cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print(f"{'degree':>6} {'alpha':>10} {'CV R² mean':>12} {'CV R² std':>11}")
results = []
for degree in [1, 2, 3]:
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        r = cv_ridge_logspace(X_train, y_train, degree, alpha, cv)
        results.append({"degree": degree, "alpha": alpha,
                        "cv_mean": r["mean"], "cv_std": r["std"]})
        print(f"{degree:>6} {alpha:>10g} {r['mean']:>+12.5f} {r['std']:>11.5f}")

results_df = pd.DataFrame(results).sort_values("cv_mean", ascending=False)
print("\nTop 5 configurations:")
print(results_df.head().to_string(index=False))

best = results_df.iloc[0]
best_degree = int(best["degree"])
best_alpha = float(best["alpha"])
print(f"\nBest: degree={best_degree}, alpha={best_alpha}, "
      f"CV R² = {best['cv_mean']:.5f} ± {best['cv_std']:.5f}")


# ---------------------------------------------------------------------------
# 6. Fit best on full train + submission
# ---------------------------------------------------------------------------
section("6. FIT BEST + SUBMISSION")
y_train_log = signed_log(y_train)
best_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=best_degree, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=best_alpha, random_state=SEED)),
])
best_pipe.fit(X_train, y_train_log)

# In-sample R² (in original space)
train_pred_log = best_pipe.predict(X_train)
train_pred = signed_exp(train_pred_log)
train_r2 = r2_score(y_train, train_pred)
print(f"Train R² (in-sample): {train_r2:.5f}")
print(f"CV R² mean         : {best['cv_mean']:.5f}")
print(f"Generalization gap : {train_r2 - best['cv_mean']:+.5f}")

# Test prediction
test_pred_log = best_pipe.predict(X_test)
test_pred = signed_exp(test_pred_log)
print(f"\nTest prediction stats: mean={test_pred.mean():.2f}, "
      f"std={test_pred.std():.2f}, "
      f"min={test_pred.min():.2f}, max={test_pred.max():.2f}")

submission = pd.DataFrame({ID_COL: test_ids, "target": test_pred})
submission = submission.set_index(ID_COL).loc[sample[ID_COL]].reset_index()

version = f"v03_ridge_logtarget_poly{best_degree}_alpha{best_alpha:g}"
out_path = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_{version}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(submission.head())


# ---------------------------------------------------------------------------
# 7. Experiments.md line
# ---------------------------------------------------------------------------
section("7. EXPERIMENTS.MD LINE")
md_line = (
    f"| 03 | (today) | Ridge + Poly(d={best_degree}) on signed-log target, "
    f"α={best_alpha:g} | {best['cv_mean']:.5f} ± {best['cv_std']:.5f} | TBD "
    f"| {out_path.name} | Target transform to handle heavy tails. |"
)
print(md_line)
