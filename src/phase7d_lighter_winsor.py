"""
Phase 7d: Push further — lighter winsor and even smaller alpha
==============================================================
v7b winner was Winsor[0.5-99.5] + Ridge(α=1).
Pattern: lighter winsor + smaller alpha → higher OOF.
Test even lighter winsor (0.1-99.9, 0.2-99.8) and α < 1.
Also asymmetric winsor (skewed target).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
SEED = 42
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]
SUBMISSION_NAME = "Baek_Seunghan"


def section(t): print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def winsor_filter(lo, hi):
    def f(X, y):
        a, b = np.percentile(y, [lo, hi])
        return X, np.clip(y, a, b)
    return f


def cv_oof_filtered(pipe_factory, X, y, cv, y_filter):
    fold_scores, oof = [], np.zeros_like(y, dtype=float)
    for tr, va in cv.split(X):
        X_tr, y_tr = y_filter(X[tr], y[tr])
        p = pipe_factory().fit(X_tr, y_tr)
        pred = p.predict(X[va])
        oof[va] = pred
        fold_scores.append(r2_score(y[va], pred))
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), float(r2_score(y, oof))


section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X_train = train[FEATURE_COLS].values
y_train = train["target"].values
X_test = test[FEATURE_COLS].values
test_ids = test["Id"].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results = []


section("LIGHTER SYMMETRIC WINSOR + SMALL ALPHA")
for lo, hi in [(0.1, 99.9), (0.2, 99.8), (0.3, 99.7), (0.5, 99.5), (0.7, 99.3)]:
    for alpha in [0.01, 0.1, 0.3, 1.0, 3.0]:
        def fac(a=alpha):
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("m", Ridge(alpha=a, random_state=SEED))])
        cvm, cvs, oof = cv_oof_filtered(fac, X_train, y_train, cv, winsor_filter(lo, hi))
        print(f"  Winsor[{lo:>4}-{hi}]+Ridge α={alpha:>5g}: CV={cvm:+.5f}, OOF={oof:+.5f}")
        results.append({"name": f"Winsor[{lo}-{hi}]+Ridge(α={alpha:g})",
                        "cv_mean": cvm, "cv_std": cvs, "oof_r2": oof,
                        "factory": fac, "y_filter": winsor_filter(lo, hi)})


section("ASYMMETRIC WINSOR (target skewed)")
# target percentiles: p0=-41008, p1=-2655, p99=+1591, p100=+69628
# Positive tail is more extreme — try clipping only one side or asymmetric
for lo, hi in [(0.5, 99.0), (1.0, 99.5), (0.3, 99.7), (0.7, 99.5),
               (0.5, 99.9), (0.1, 99.5)]:
    for alpha in [0.1, 1, 10]:
        def fac(a=alpha):
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("m", Ridge(alpha=a, random_state=SEED))])
        cvm, cvs, oof = cv_oof_filtered(fac, X_train, y_train, cv, winsor_filter(lo, hi))
        print(f"  Asym[{lo:>4}-{hi}]+Ridge α={alpha:>4g}: CV={cvm:+.5f}, OOF={oof:+.5f}")
        results.append({"name": f"Asym[{lo}-{hi}]+Ridge(α={alpha:g})",
                        "cv_mean": cvm, "cv_std": cvs, "oof_r2": oof,
                        "factory": fac, "y_filter": winsor_filter(lo, hi)})


section("TOP 15 by OOF R²")
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("factory", "y_filter")}
                   for r in results]).sort_values("oof_r2", ascending=False).reset_index(drop=True)
print(df.head(15).to_string(index=False))

current_best_oof = 0.03266  # v7b
print(f"\nCurrent best (v7b): OOF +{current_best_oof:.5f}, LB +0.04843")
print(f"Best here         : OOF {df.iloc[0]['oof_r2']:+.5f}")
print(f"Improvement       : {df.iloc[0]['oof_r2'] - current_best_oof:+.5f}")

if df.iloc[0]['oof_r2'] > current_best_oof:
    best_name = df.iloc[0]["name"]
    best = [r for r in results if r["name"] == best_name][0]
    expected_lb = best["oof_r2"] + 0.018  # linear-model calibration
    print(f"  → Expected LB ≈ +{expected_lb:.4f}  (vs current v7b +0.04843)")

    section("BUILD CANDIDATE SUBMISSION")
    pipe = best["factory"]()
    X_fit, y_fit = best["y_filter"](X_train, y_train)
    pipe.fit(X_fit, y_fit)
    pred = pipe.predict(X_test)
    print(f"Test pred: mean={pred.mean():.2f}, std={pred.std():.2f}, "
          f"min={pred.min():.2f}, max={pred.max():.2f}")
    sub = pd.DataFrame({"Id": test_ids, "target": pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    short = (best["name"].lower().replace(" ", "").replace("=", "")
             .replace("(", "_").replace(")", "").replace(",", "_")
             .replace("α", "a").replace("[", "").replace("]", "")
             .replace("-", "to").replace(".", ""))
    out = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_v07d_{short}.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
else:
    print("\nNo improvement — v7b stays best.")
