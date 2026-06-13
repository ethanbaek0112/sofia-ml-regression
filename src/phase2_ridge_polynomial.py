"""
Phase 2: Polynomial Regression + Ridge (L2) — Lec 07
=====================================================

Hypothesis
----------
Plain LSM failed (CV R² ≈ -0.46) because:
  (a) features have weak linear correlations with target (max |r| = 0.25)
  (b) target is heavy-tailed (skew=13.6) — LSM is "highly sensitive to outliers"
      per Lec 07 slide 11.

Lec 07 slide 10 ("Polynomial Regression") says we can generate new features by
applying non-linear transforms (pairwise products, exponentiation, sqrt).
This captures non-linear dependencies while still using a linear model in the
expanded feature space.

But polynomial expansion creates many features (degree 2 on 15 features = 135),
which causes overfitting. The fix per Lec 07 slide 14 is L2 regularization:
  L = MSE + α/2 * ||w||^2     (Ridge Regression)
α is a hyperparameter we tune via cross-validation (Lec 02).

Standard scaling note
---------------------
We apply StandardScaler before Ridge because the L2 penalty depends on the
scale of weights, which depends on the scale of features. Without scaling,
features with large variance (e.g., x5 std=15) would dominate the penalty
relative to small-variance features (e.g., x3 std=2).
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


def build_pipeline(degree: int) -> Pipeline:
    """Median imputer -> Polynomial features -> Standard scaler -> Ridge."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=SEED)),
    ])


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
# 2. Compare polynomial degrees (sanity check)
# ------------------------------------------------------------------
section("2. DEGREE COMPARISON (Ridge α=1.0 default, just to see effect of poly)")
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for degree in [1, 2, 3]:
    pipe = build_pipeline(degree)
    scores = cross_val_score(pipe, X_train, y_train,
                             scoring="r2", cv=cv, n_jobs=-1)
    n_features_after_poly = (
        pipe.named_steps["poly"]
        .fit(SimpleImputer(strategy="median").fit_transform(X_train))
        .n_output_features_
    )
    print(f"  degree={degree}: {n_features_after_poly:>4} features, "
          f"CV R² = {scores.mean():+.5f} ± {scores.std():.5f}")


# ---------------------------------------------------------------------------
# 3. Grid search over (degree, alpha) — Lec 02 hyperparameter tuning
# ---------------------------------------------------------------------------
section("3. GRID SEARCH OVER (degree, α)")
# Build a single pipeline; GridSearchCV will vary params via name__param syntax.
pipe = build_pipeline(degree=2)  # degree is overridden by grid

param_grid = {
    "poly__degree": [2, 3],
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
}
grid = GridSearchCV(
    pipe, param_grid=param_grid,
    scoring="r2", cv=cv, n_jobs=-1, verbose=1, refit=True,
)
grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)[
    ["param_poly__degree", "param_model__alpha",
     "mean_test_score", "std_test_score"]
].sort_values("mean_test_score", ascending=False)
print("\nTop 10 configurations by CV R²:")
print(results.head(10).to_string(index=False))

print(f"\nBest params: {grid.best_params_}")
print(f"Best CV R² : {grid.best_score_:.5f}")


# ---------------------------------------------------------------------------
# 4. Fit best pipeline on full training data + diagnostics
# ---------------------------------------------------------------------------
section("4. FIT BEST + DIAGNOSTICS")
best = grid.best_estimator_
best.fit(X_train, y_train)

train_pred = best.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
print(f"Train R² (in-sample, optimistic): {train_r2:.5f}")
print(f"CV R² mean                     : {grid.best_score_:.5f}")
print(f"Generalization gap               : {train_r2 - grid.best_score_:+.5f}")
print("=> Big gap = overfitting; small gap = healthy generalization (Lec 03)")


# ---------------------------------------------------------------------------
# 5. Predict test & save submission
# ---------------------------------------------------------------------------
section("5. PREDICT TEST & SAVE SUBMISSION")
test_pred = best.predict(X_test)
print(f"Test prediction stats: mean={test_pred.mean():.2f}, "
      f"std={test_pred.std():.2f}, "
      f"min={test_pred.min():.2f}, max={test_pred.max():.2f}")

submission = pd.DataFrame({ID_COL: test_ids, "target": test_pred})
submission = submission.set_index(ID_COL).loc[sample[ID_COL]].reset_index()

best_degree = grid.best_params_["poly__degree"]
best_alpha = grid.best_params_["model__alpha"]
version = f"v02_ridge_poly{best_degree}_alpha{best_alpha:g}"
out_path = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_{version}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(submission.head())


# ---------------------------------------------------------------------------
# 6. Experiments.md line
# ---------------------------------------------------------------------------
section("6. EXPERIMENTS.MD LINE")
md_line = (
    f"| 02 | (today) | Ridge + PolynomialFeatures(degree={best_degree}) "
    f"+ StandardScaler, α={best_alpha:g} "
    f"| {grid.best_score_:.5f} ± "
    f"{results.iloc[0]['std_test_score']:.5f} "
    f"| TBD | {out_path.name} "
    f"| Polynomial captures non-linearity, Ridge handles overfit. |"
)
print(md_line)
