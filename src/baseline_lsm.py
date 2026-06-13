"""
Baseline: Linear Regression (Least Squares Method) — Lec 07
============================================================

This is the simplest model from the curriculum:
- ML estimator under Gaussian noise assumption (Lec 07 slide 6)
- Closed-form analytical solution: w = (X^T X)^-1 X^T y (Lec 07 slide 9)
- No regularization, no feature engineering

Pipeline:
1. Impute missing values with column median (simple, safe default)
2. Fit LinearRegression on full training data
3. Evaluate via 5-fold cross-validation (Lec 02)
4. Generate first submission CSV
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5

FEATURE_COLS = [f"x{i}" for i in range(15)]
TARGET_COL = "target"
ID_COL = "Id"

SUBMISSION_NAME = "Baek_Seunghan"
VERSION = "v01_baseline_lsm"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
section("1. LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"train: {train.shape}, test: {test.shape}")

X_train = train[FEATURE_COLS].values
y_train = train[TARGET_COL].values
X_test = test[FEATURE_COLS].values
test_ids = test[ID_COL].values


# ---------------------------------------------------------------------------
# 2. Build pipeline
# ---------------------------------------------------------------------------
section("2. BUILD PIPELINE")
# Median imputer is chosen because target is heavy-tailed: median is more robust
# than mean to extreme values (and we expect feature distributions to be similar).
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression()),
])
print(pipeline)


# ---------------------------------------------------------------------------
# 3. Cross-validation (Lec 02): the trustworthy local estimate
# ---------------------------------------------------------------------------
section("3. 5-FOLD CROSS-VALIDATION (R²)")
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    scoring="r2", cv=cv, n_jobs=-1,
)
print(f"Fold R² scores: {np.round(cv_scores, 5)}")
print(f"CV R² mean     : {cv_scores.mean():.5f}")
print(f"CV R² std      : {cv_scores.std():.5f}")
print(f"CV R² min/max  : {cv_scores.min():.5f} / {cv_scores.max():.5f}")


# ---------------------------------------------------------------------------
# 4. Fit on full training data
# ---------------------------------------------------------------------------
section("4. FIT ON FULL TRAIN")
pipeline.fit(X_train, y_train)
train_pred = pipeline.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
print(f"Train R² (in-sample, optimistic): {train_r2:.5f}")
print(f"Compared to CV R² mean         : {cv_scores.mean():.5f}")
print(f"=> Generalization gap: {train_r2 - cv_scores.mean():+.5f}")

# Coefficients for interpretation (Lec 07: Linear Regression is well-interpretable)
lr = pipeline.named_steps["model"]
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "weight": lr.coef_,
}).reindex(np.abs(lr.coef_).argsort()[::-1])
print(f"\nIntercept (w0): {lr.intercept_:.4f}")
print("Top 5 features by |weight|:")
print(coef_df.head().to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Predict test + save submission
# ---------------------------------------------------------------------------
section("5. PREDICT TEST & SAVE SUBMISSION")
test_pred = pipeline.predict(X_test)
print(f"Test predictions stats:")
print(f"  mean : {test_pred.mean():.4f}")
print(f"  std  : {test_pred.std():.4f}")
print(f"  min  : {test_pred.min():.4f}")
print(f"  max  : {test_pred.max():.4f}")

submission = pd.DataFrame({ID_COL: test_ids, "target": test_pred})
# Match sample_submission ID order
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
submission = submission.set_index(ID_COL).loc[sample[ID_COL]].reset_index()

out_path = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_{VERSION}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Submission head:\n{submission.head()}")
print(f"Rows: {len(submission)} (expected {len(sample)})")


# ---------------------------------------------------------------------------
# 6. Result line for experiments.md
# ---------------------------------------------------------------------------
section("6. EXPERIMENTS.MD LINE")
md_line = (
    f"| 01 | (today) | Linear Regression (LSM) + median imputation "
    f"| {cv_scores.mean():.5f} ± {cv_scores.std():.5f} | TBD | "
    f"{out_path.name} | Baseline. No FE, no regularization. |"
)
print(md_line)
