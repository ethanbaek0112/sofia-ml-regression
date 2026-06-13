"""
Phase 6: Ensemble submissions
==============================

Phase 5 finding: averaging best ElasticNet (+0.02137) with best k-NN (+0.02105)
gave +0.02351 — a real improvement. Why? They make different errors on different
examples (decorrelated errors average out — a textbook ensemble benefit).

Polynomial features and Lec07 transforms with heavy regularization did NOT help
beyond Phase 4. The 3 model families really do cap around +0.021 individually.

Final attempts:
  v6a: 2-way ensemble (ElasticNet + k-NN), 50/50 average
  v6b: 3-way ensemble (Ridge + ElasticNet + k-NN), 33/33/33 average
  v6c: Weighted ensemble (optimize weights via CV)
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"

SEED = 42
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]
SUBMISSION_NAME = "Baek_Seunghan"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def make_en() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=10.0, l1_ratio=0.5,
                              random_state=SEED, max_iter=20000)),
    ])


def make_ridge() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10_000.0, random_state=SEED)),
    ])


def make_knn() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=200, weights="distance",
                                      n_jobs=-1)),
    ])


def get_oof_predictions(pipe_factory, X, y, cv: KFold):
    """Out-of-fold predictions for ensemble weight tuning."""
    oof = np.zeros_like(y, dtype=float)
    for tr_idx, va_idx in cv.split(X):
        pipe = pipe_factory()
        pipe.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = pipe.predict(X[va_idx])
    return oof


# Load
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X_train = train[FEATURE_COLS].values
y_train = train[TARGET_COL := "target"].values
X_test = test[FEATURE_COLS].values
test_ids = test["Id"].values
print(f"train: {train.shape}, test: {test.shape}")

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# Get OOF predictions for each model
section("GET OUT-OF-FOLD PREDICTIONS")
oof_en = get_oof_predictions(make_en, X_train, y_train, cv)
oof_ridge = get_oof_predictions(make_ridge, X_train, y_train, cv)
oof_knn = get_oof_predictions(make_knn, X_train, y_train, cv)

print(f"OOF R²  ElasticNet : {r2_score(y_train, oof_en):+.5f}")
print(f"OOF R²  Ridge      : {r2_score(y_train, oof_ridge):+.5f}")
print(f"OOF R²  k-NN       : {r2_score(y_train, oof_knn):+.5f}")


# Simple averages
section("SIMPLE AVERAGES")
avg_en_knn = 0.5 * oof_en + 0.5 * oof_knn
avg_3 = (oof_en + oof_ridge + oof_knn) / 3.0
print(f"OOF R² (EN + k-NN avg)            : {r2_score(y_train, avg_en_knn):+.5f}")
print(f"OOF R² (Ridge + EN + k-NN avg, 1/3 each): {r2_score(y_train, avg_3):+.5f}")


# Weighted ensemble (small grid)
section("WEIGHT GRID (sum = 1)")
best_w = None
best_r2 = -np.inf
for w_en in np.arange(0.0, 1.01, 0.1):
    for w_ridge in np.arange(0.0, 1.01 - w_en, 0.1):
        w_knn = 1.0 - w_en - w_ridge
        if w_knn < -1e-9:
            continue
        pred = w_en * oof_en + w_ridge * oof_ridge + w_knn * oof_knn
        r2 = r2_score(y_train, pred)
        if r2 > best_r2:
            best_r2 = r2
            best_w = (w_en, w_ridge, w_knn)

print(f"Best weights (EN, Ridge, k-NN) = {tuple(round(w, 2) for w in best_w)}")
print(f"Best OOF R² = {best_r2:+.5f}")


# Build all three submissions
section("BUILD SUBMISSIONS")
# Fit all three models on FULL training data, then ensemble
en_full = make_en().fit(X_train, y_train)
ridge_full = make_ridge().fit(X_train, y_train)
knn_full = make_knn().fit(X_train, y_train)

pred_en = en_full.predict(X_test)
pred_ridge = ridge_full.predict(X_test)
pred_knn = knn_full.predict(X_test)


def save_submission(pred: np.ndarray, version: str, oof_score: float) -> Path:
    sub = pd.DataFrame({"Id": test_ids, "target": pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    path = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_{version}.csv"
    sub.to_csv(path, index=False)
    print(f"  {version:<35}  OOF R² = {oof_score:+.5f}  →  {path.name}")
    return path


# v6a: simple 50/50 EN + k-NN
pred_v6a = 0.5 * pred_en + 0.5 * pred_knn
save_submission(pred_v6a, "v06a_ens_en_knn_50_50",
                r2_score(y_train, avg_en_knn))

# v6b: 3-way 1/3 each
pred_v6b = (pred_en + pred_ridge + pred_knn) / 3.0
save_submission(pred_v6b, "v06b_ens_3way_equal",
                r2_score(y_train, avg_3))

# v6c: best weighted ensemble
w_en, w_ridge, w_knn = best_w
pred_v6c = w_en * pred_en + w_ridge * pred_ridge + w_knn * pred_knn
short = f"en{w_en:.1f}_ridge{w_ridge:.1f}_knn{w_knn:.1f}".replace(".", "")
save_submission(pred_v6c, f"v06c_ens_weighted_{short}", best_r2)

section("SUMMARY")
print(f"Phase 4 best     (v4): +0.02137  ElasticNet alone")
print(f"v6a EN+kNN 50/50    : {r2_score(y_train, avg_en_knn):+.5f}")
print(f"v6b 3-way equal     : {r2_score(y_train, avg_3):+.5f}")
print(f"v6c best weighted   : {best_r2:+.5f}  weights={tuple(round(w, 2) for w in best_w)}")
print(f"\nKaggle baseline target: +0.04599")
print(f"Best gap remaining   : {0.04599 - best_r2:+.5f}")
