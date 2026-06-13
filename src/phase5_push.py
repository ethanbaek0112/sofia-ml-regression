"""
Phase 5: Push past the baseline (target = beat 0.04599 on Kaggle LB)
=====================================================================

Context
-------
Phase 4 found that raw features cap out at CV R² ≈ +0.021 across three model
families (Ridge, ElasticNet, k-NN). Public LB baseline is 0.04599.
Gap to close: ~+0.025 R².

Lec 07 slide 10 explicitly lists feature transformations we can use:
  - pairwise products
  - exponentiation (squares)
  - sqrt of absolute value
These create new features in a non-linear way while keeping the model linear.

Phase 2's polynomial Ridge failed because alpha only went up to 1000.
Phase 4 found alpha=10000+ is what works for raw features.
=> Crucial untested combination: Polynomial features + alpha >= 10000.

Experiments (all within curriculum)
-----------------------------------
A. Poly(d=2) + Ridge, alpha sweep including 10k+
B. Poly(d=2) + ElasticNet, alpha sweep including 10+
C. Manual Lec07-style transforms (sqrt, square) + Ridge
D. Average of best ElasticNet and best k-NN (simple ensemble)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

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
# A. Polynomial(d=2) + Ridge with EXTENDED alpha (untested before)
# ---------------------------------------------------------------------------
section("A. POLYNOMIAL(d=2) + RIDGE — extended α range")
for alpha in [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha, random_state=SEED)),
    ])
    mean, std = cv_score(pipe, X_train, y_train, cv)
    print(f"  Poly(d=2)+Ridge α={alpha:>10g}: CV R² = {mean:+.5f} ± {std:.5f}")
    results.append({
        "name": f"Poly2+Ridge(α={alpha:g})",
        "cv_mean": mean, "cv_std": std,
        "factory": lambda a=alpha: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=a, random_state=SEED)),
        ]),
    })


# ---------------------------------------------------------------------------
# B. Polynomial(d=2) + ElasticNet
# ---------------------------------------------------------------------------
section("B. POLYNOMIAL(d=2) + ELASTICNET — α and l1_ratio sweep")
for alpha in [1.0, 10.0, 100.0, 1000.0]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                  random_state=SEED, max_iter=30000)),
        ])
        mean, std = cv_score(pipe, X_train, y_train, cv)
        print(f"  Poly(d=2)+EN α={alpha:>6g}, l1_ratio={l1_ratio:>3}: "
              f"CV R² = {mean:+.5f} ± {std:.5f}")
        results.append({
            "name": f"Poly2+EN(α={alpha:g},l1={l1_ratio})",
            "cv_mean": mean, "cv_std": std,
            "factory": lambda a=alpha, r=l1_ratio: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=a, l1_ratio=r,
                                     random_state=SEED, max_iter=30000)),
            ]),
        })


# ---------------------------------------------------------------------------
# C. Lec 07 slide 10 manual transforms (sqrt|x| and x^2) + Ridge
# ---------------------------------------------------------------------------
section("C. MANUAL TRANSFORMS (sqrt|x|, x²) + RIDGE")


def lec07_transform(X: np.ndarray) -> np.ndarray:
    """Concat original, signed-sqrt, and squared features (Lec 07 slide 10)."""
    sqrt_abs = np.sign(X) * np.sqrt(np.abs(X))
    squared = X ** 2
    return np.hstack([X, sqrt_abs, squared])


for alpha in [10, 100, 1000, 10_000, 100_000]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lec07_xform", FunctionTransformer(lec07_transform)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha, random_state=SEED)),
    ])
    mean, std = cv_score(pipe, X_train, y_train, cv)
    print(f"  Lec07Xform+Ridge α={alpha:>8g}: CV R² = {mean:+.5f} ± {std:.5f}")
    results.append({
        "name": f"Lec07Xform+Ridge(α={alpha:g})",
        "cv_mean": mean, "cv_std": std,
        "factory": lambda a=alpha: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("lec07_xform", FunctionTransformer(lec07_transform)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=a, random_state=SEED)),
        ]),
    })


# ---------------------------------------------------------------------------
# D. Simple average ensemble (best ElasticNet + best k-NN from Phase 4)
# ---------------------------------------------------------------------------
section("D. ENSEMBLE: average of best ElasticNet + best k-NN")
en_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=10.0, l1_ratio=0.5,
                          random_state=SEED, max_iter=20000)),
])
knn_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=200, weights="distance",
                                  n_jobs=-1)),
])

# Custom CV: average predictions from both models per fold
fold_scores = []
for tr_idx, va_idx in cv.split(X_train):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    en_pipe.fit(X_tr, y_tr)
    knn_pipe.fit(X_tr, y_tr)
    pred = 0.5 * en_pipe.predict(X_va) + 0.5 * knn_pipe.predict(X_va)
    fold_scores.append(r2_score(y_va, pred))
en_knn_mean, en_knn_std = float(np.mean(fold_scores)), float(np.std(fold_scores))
print(f"  ElasticNet + k-NN avg: CV R² = {en_knn_mean:+.5f} ± {en_knn_std:.5f}")


# ---------------------------------------------------------------------------
# 6. Ranking
# ---------------------------------------------------------------------------
section("RANKING (top 10)")
results_df = pd.DataFrame(results).sort_values("cv_mean", ascending=False)
print(results_df.drop(columns=["factory"]).head(10).to_string(index=False))

best = results_df.iloc[0]
best_name = best["name"]
best_cv = best["cv_mean"]
print(f"\nWinner: {best_name}  →  CV R² = {best_cv:+.5f}")
print(f"Phase 4 winner was +0.02137 — improvement: "
      f"{best_cv - 0.02137:+.5f}")
print(f"Kaggle baseline target: +0.04599 — "
      f"gap remaining: {0.04599 - best_cv:+.5f}")


# ---------------------------------------------------------------------------
# 7. Fit best + submission
# ---------------------------------------------------------------------------
section("FIT BEST + SUBMISSION")
best_pipe = best["factory"]()
best_pipe.fit(X_train, y_train)
train_pred = best_pipe.predict(X_train)
print(f"Train R² (in-sample): {r2_score(y_train, train_pred):+.5f}")
print(f"CV R² mean         : {best_cv:+.5f}")
print(f"Gap (train - CV)   : {r2_score(y_train, train_pred) - best_cv:+.5f}")

test_pred = best_pipe.predict(X_test)
print(f"\nTest pred stats: mean={test_pred.mean():.2f}, "
      f"std={test_pred.std():.2f}, "
      f"min={test_pred.min():.2f}, max={test_pred.max():.2f}")

submission = pd.DataFrame({ID_COL: test_ids, "target": test_pred})
submission = submission.set_index(ID_COL).loc[sample[ID_COL]].reset_index()

short = (best_name.lower()
         .replace(" ", "")
         .replace("=", "")
         .replace("(", "_")
         .replace(")", "")
         .replace(",", "_")
         .replace("α", "a")
         .replace(".", ""))
version = f"v05_{short}"
out_path = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_{version}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(submission.head())

# Save the full ranked table for inspection
results_df.drop(columns=["factory"]).to_csv(
    ROOT / "docs" / "phase5_results.csv", index=False)
print(f"\nFull results table: {ROOT / 'docs' / 'phase5_results.csv'}")
