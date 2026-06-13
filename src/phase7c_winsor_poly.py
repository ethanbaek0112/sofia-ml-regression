"""
Phase 7c: Winsor + Polynomial features
=======================================
Hypothesis: With Winsorizing removing the extreme target outliers, polynomial
feature expansion may finally help (instead of amplifying outliers).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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


section("WINSOR + POLY + RIDGE/ELASTICNET")
for lo, hi in [(0.5, 99.5), (1, 99)]:
    for degree in [2]:
        for alpha in [1, 10, 100, 1000, 10_000]:
            def fac(d=degree, a=alpha):
                return Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                    ("sc", StandardScaler()),
                    ("m", Ridge(alpha=a, random_state=SEED)),
                ])
            cvm, cvs, oof = cv_oof_filtered(fac, X_train, y_train, cv,
                                             winsor_filter(lo, hi))
            print(f"  Winsor[{lo:>4}-{hi}]+Poly{degree}+Ridge α={alpha:>5g}: "
                  f"CV={cvm:+.5f}, OOF={oof:+.5f}")
            results.append({"name": f"Winsor[{lo}-{hi}]+Poly{degree}+Ridge(α={alpha:g})",
                            "cv_mean": cvm, "cv_std": cvs, "oof_r2": oof,
                            "factory": fac, "y_filter": winsor_filter(lo, hi)})


section("WINSOR + POLY + ELASTICNET")
for lo, hi in [(0.5, 99.5), (1, 99)]:
    for alpha in [1, 10, 100]:
        for l1 in [0.1, 0.5, 0.9]:
            def fac(a=alpha, r=l1):
                return Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("sc", StandardScaler()),
                    ("m", ElasticNet(alpha=a, l1_ratio=r, random_state=SEED,
                                      max_iter=30000)),
                ])
            cvm, cvs, oof = cv_oof_filtered(fac, X_train, y_train, cv,
                                             winsor_filter(lo, hi))
            print(f"  Winsor[{lo:>4}-{hi}]+Poly2+EN α={alpha:>4g} l1={l1}: "
                  f"CV={cvm:+.5f}, OOF={oof:+.5f}")
            results.append({"name": f"Winsor[{lo}-{hi}]+Poly2+EN(α={alpha:g},l1={l1})",
                            "cv_mean": cvm, "cv_std": cvs, "oof_r2": oof,
                            "factory": fac, "y_filter": winsor_filter(lo, hi)})


section("TOP 10 by OOF R²")
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("factory", "y_filter")}
                   for r in results]).sort_values("oof_r2", ascending=False).reset_index(drop=True)
print(df.head(10).to_string(index=False))

# Compare to current best
current_best_oof = 0.03266  # Winsor[0.5-99.5]+Ridge(α=1)
print(f"\nCurrent best (v7b): OOF +{current_best_oof:.5f}")
print(f"Best here         : OOF {df.iloc[0]['oof_r2']:+.5f}")
print(f"Improvement       : {df.iloc[0]['oof_r2'] - current_best_oof:+.5f}")

best_name = df.iloc[0]["name"]
best = [r for r in results if r["name"] == best_name][0]

if best["oof_r2"] > current_best_oof:
    section("NEW BEST — BUILD SUBMISSION")
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
    out = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_v07c_{short}.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
else:
    print("\nNo improvement — v7b remains our best.")
