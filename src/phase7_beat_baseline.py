"""
Phase 7: Beat the baseline (target = +0.04599 LB)
==================================================

Key insight from v4 Kaggle result:
  CV R² = +0.0214  →  Public LB R² = +0.0320
  Calibration gap: +0.011 (CV underestimates because of fold-of-doom)

To beat baseline (+0.04599), we need a model whose CV (or OOF) R² ≈ +0.035+.

Untested ideas (all curriculum-aligned):
A. ElasticNet on raw features with α > 100 (we only tested up to 100)
B. k-NN with larger k (we only tested up to 200, dataset has 2500)
C. Winsorizing: clip extreme target values in TRAINING data only
   (a standard "data cleaning" technique that helps with outlier-sensitive
   linear models; Lec 07 acknowledges outlier sensitivity as a weakness)
D. Trimming: drop extreme target rows from training entirely

For each, we compute BOTH cv_score (mean of per-fold R²) and OOF R²
(concat predictions, score once) — the latter is closer to Kaggle.
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


def cv_and_oof(pipe_factory, X, y, cv: KFold) -> tuple[float, float, float, np.ndarray]:
    """Return cv_mean, cv_std, oof_r2, oof_preds."""
    fold_scores = []
    oof = np.zeros_like(y, dtype=float)
    for tr_idx, va_idx in cv.split(X):
        pipe = pipe_factory()
        pipe.fit(X[tr_idx], y[tr_idx])
        pred = pipe.predict(X[va_idx])
        oof[va_idx] = pred
        fold_scores.append(r2_score(y[va_idx], pred))
    return (float(np.mean(fold_scores)),
            float(np.std(fold_scores)),
            float(r2_score(y, oof)),
            oof)


def cv_and_oof_with_y_filter(pipe_factory, X, y, cv: KFold,
                              y_filter_fn) -> tuple[float, float, float, np.ndarray]:
    """Same as cv_and_oof but applies y_filter_fn to training y only."""
    fold_scores = []
    oof = np.zeros_like(y, dtype=float)
    for tr_idx, va_idx in cv.split(X):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        # Apply training-only filter (e.g., trim outliers from train)
        X_tr_filt, y_tr_filt = y_filter_fn(X_tr, y_tr)
        pipe = pipe_factory()
        pipe.fit(X_tr_filt, y_tr_filt)
        pred = pipe.predict(X[va_idx])
        oof[va_idx] = pred
        fold_scores.append(r2_score(y[va_idx], pred))
    return (float(np.mean(fold_scores)),
            float(np.std(fold_scores)),
            float(r2_score(y, oof)),
            oof)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X_train = train[FEATURE_COLS].values
y_train = train["target"].values
X_test = test[FEATURE_COLS].values
test_ids = test["Id"].values
print(f"train: {train.shape}, test: {test.shape}")

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results: list[dict] = []


# ---------------------------------------------------------------------------
# A. ElasticNet raw features with HIGH alpha (untested >100)
# ---------------------------------------------------------------------------
section("A. ELASTICNET (raw) — HIGH α SWEEP")
for alpha in [100, 300, 1000, 3000, 10_000, 30_000]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        def factory(a=alpha, r=l1_ratio):
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=a, l1_ratio=r,
                                      random_state=SEED, max_iter=30000)),
            ])
        cv_mean, cv_std, oof_r2, _ = cv_and_oof(factory, X_train, y_train, cv)
        print(f"  EN α={alpha:>6g}, l1={l1_ratio}: "
              f"CV={cv_mean:+.5f} ± {cv_std:.5f},  OOF={oof_r2:+.5f}")
        results.append({"name": f"EN(α={alpha:g},l1={l1_ratio})",
                        "cv_mean": cv_mean, "cv_std": cv_std,
                        "oof_r2": oof_r2, "factory": factory})


# ---------------------------------------------------------------------------
# B. k-NN with larger k
# ---------------------------------------------------------------------------
section("B. k-NN — LARGER k SWEEP")
for k in [200, 300, 500, 800, 1200, 1800]:
    for weights in ["uniform", "distance"]:
        def factory(kk=k, ww=weights):
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=kk, weights=ww,
                                              n_jobs=-1)),
            ])
        cv_mean, cv_std, oof_r2, _ = cv_and_oof(factory, X_train, y_train, cv)
        print(f"  k-NN k={k:>4}, w={weights:<8}: "
              f"CV={cv_mean:+.5f} ± {cv_std:.5f},  OOF={oof_r2:+.5f}")
        results.append({"name": f"kNN(k={k},{weights})",
                        "cv_mean": cv_mean, "cv_std": cv_std,
                        "oof_r2": oof_r2, "factory": factory})


# ---------------------------------------------------------------------------
# C. Winsorizing TRAINING target (clip extreme y values in training data)
# ---------------------------------------------------------------------------
section("C. WINSORIZE training target — Ridge/ElasticNet")


def make_winsorize_filter(lo_pct: float, hi_pct: float):
    def filt(X, y):
        lo, hi = np.percentile(y, [lo_pct, hi_pct])
        y_clipped = np.clip(y, lo, hi)
        return X, y_clipped
    return filt


def make_trim_filter(lo_pct: float, hi_pct: float):
    def filt(X, y):
        lo, hi = np.percentile(y, [lo_pct, hi_pct])
        mask = (y >= lo) & (y <= hi)
        return X[mask], y[mask]
    return filt


for lo, hi in [(1, 99), (2.5, 97.5), (5, 95), (10, 90)]:
    for alpha in [10, 100, 1000, 10_000]:
        def factory(a=alpha):
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=a, random_state=SEED)),
            ])
        winsorize = make_winsorize_filter(lo, hi)
        cv_mean, cv_std, oof_r2, _ = cv_and_oof_with_y_filter(
            factory, X_train, y_train, cv, winsorize)
        print(f"  Winsor[{lo:>4}-{hi}]+Ridge α={alpha:>5g}: "
              f"CV={cv_mean:+.5f} ± {cv_std:.5f},  OOF={oof_r2:+.5f}")
        results.append({
            "name": f"Winsor[{lo}-{hi}]+Ridge(α={alpha:g})",
            "cv_mean": cv_mean, "cv_std": cv_std,
            "oof_r2": oof_r2,
            "factory": factory,
            "y_filter": winsorize,
        })


# ---------------------------------------------------------------------------
# D. Trimming: DROP extreme rows from training
# ---------------------------------------------------------------------------
section("D. TRIM training (drop outlier rows) — Ridge/ElasticNet")
for lo, hi in [(1, 99), (2.5, 97.5), (5, 95)]:
    for alpha in [10, 100, 1000, 10_000]:
        def factory(a=alpha):
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=a, random_state=SEED)),
            ])
        trim = make_trim_filter(lo, hi)
        cv_mean, cv_std, oof_r2, _ = cv_and_oof_with_y_filter(
            factory, X_train, y_train, cv, trim)
        print(f"  Trim[{lo:>4}-{hi}]+Ridge α={alpha:>5g}: "
              f"CV={cv_mean:+.5f} ± {cv_std:.5f},  OOF={oof_r2:+.5f}")
        results.append({
            "name": f"Trim[{lo}-{hi}]+Ridge(α={alpha:g})",
            "cv_mean": cv_mean, "cv_std": cv_std,
            "oof_r2": oof_r2,
            "factory": factory,
            "y_filter": trim,
        })


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
section("RANKING by OOF R² (top 15)")
results_df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("factory", "y_filter")}
                           for r in results])
results_df = results_df.sort_values("oof_r2", ascending=False).reset_index(drop=True)
print(results_df.head(15).to_string(index=False))

best_idx = results_df.index[0]
best = results[
    [i for i, r in enumerate(results) if r["name"] == results_df.iloc[0]["name"]][0]
]
print(f"\nWinner: {best['name']}")
print(f"  CV    R² = {best['cv_mean']:+.5f} ± {best['cv_std']:.5f}")
print(f"  OOF   R² = {best['oof_r2']:+.5f}")
print(f"  Expected LB (CV + 0.011 calibration) ≈ {best['cv_mean'] + 0.011:+.4f}")
print(f"  Expected LB (OOF + 0.011 calibration) ≈ {best['oof_r2'] + 0.011:+.4f}")
print(f"  Baseline target: +0.04599")


# ---------------------------------------------------------------------------
# Fit best + submission
# ---------------------------------------------------------------------------
section("FIT BEST + SUBMISSION")
best_pipe = best["factory"]()
if "y_filter" in best:
    X_fit, y_fit = best["y_filter"](X_train, y_train)
    print(f"Using filter: train shape {X_train.shape} -> {X_fit.shape}")
else:
    X_fit, y_fit = X_train, y_train

best_pipe.fit(X_fit, y_fit)
test_pred = best_pipe.predict(X_test)
print(f"Test pred stats: mean={test_pred.mean():.2f}, std={test_pred.std():.2f}, "
      f"min={test_pred.min():.2f}, max={test_pred.max():.2f}")

submission = pd.DataFrame({"Id": test_ids, "target": test_pred})
submission = submission.set_index("Id").loc[sample["Id"]].reset_index()

short = (best["name"].lower()
         .replace(" ", "").replace("=", "").replace("(", "_")
         .replace(")", "").replace(",", "_").replace("α", "a")
         .replace("[", "").replace("]", "").replace("-", "to")
         .replace(".", ""))
version = f"v07_{short}"
out_path = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_{version}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(submission.head())

results_df.to_csv(ROOT / "docs" / "phase7_results.csv", index=False)
print(f"\nFull results: {ROOT / 'docs' / 'phase7_results.csv'}")
