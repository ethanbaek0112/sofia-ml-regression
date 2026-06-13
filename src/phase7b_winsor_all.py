"""
Phase 7b: Winsorizing applied to ALL model families
====================================================
Phase 7 found Winsor[1-99] + Ridge(α=10) gives OOF +0.02288.
Now apply Winsorizing to ElasticNet, k-NN, and try finer percentile bands.
"""
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


section("FINE-GRAINED WINSOR + RIDGE")
for lo, hi in [(0.5, 99.5), (1, 99), (1.5, 98.5), (2, 98)]:
    for alpha in [1, 3, 10, 30, 100]:
        def fac(a=alpha):
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("m", Ridge(alpha=a, random_state=SEED))])
        cvm, cvs, oof = cv_oof_filtered(fac, X_train, y_train, cv, winsor_filter(lo, hi))
        print(f"  Winsor[{lo:>4}-{hi}]+Ridge α={alpha:>4g}: CV={cvm:+.5f}, OOF={oof:+.5f}")
        results.append({"name": f"Winsor[{lo}-{hi}]+Ridge(α={alpha:g})",
                        "cv_mean": cvm, "cv_std": cvs, "oof_r2": oof,
                        "factory": fac, "y_filter": winsor_filter(lo, hi)})


section("WINSOR + ELASTICNET")
for lo, hi in [(0.5, 99.5), (1, 99), (1.5, 98.5)]:
    for alpha in [1, 10, 100]:
        for l1 in [0.1, 0.5, 0.9]:
            def fac(a=alpha, r=l1):
                return Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("sc", StandardScaler()),
                                 ("m", ElasticNet(alpha=a, l1_ratio=r,
                                                  random_state=SEED, max_iter=30000))])
            cvm, cvs, oof = cv_oof_filtered(fac, X_train, y_train, cv, winsor_filter(lo, hi))
            print(f"  Winsor[{lo:>4}-{hi}]+EN α={alpha:>4g} l1={l1}: "
                  f"CV={cvm:+.5f}, OOF={oof:+.5f}")
            results.append({"name": f"Winsor[{lo}-{hi}]+EN(α={alpha:g},l1={l1})",
                            "cv_mean": cvm, "cv_std": cvs, "oof_r2": oof,
                            "factory": fac, "y_filter": winsor_filter(lo, hi)})


section("WINSOR + k-NN")
for lo, hi in [(0.5, 99.5), (1, 99), (1.5, 98.5)]:
    for k in [100, 200, 300, 500]:
        for w in ["uniform", "distance"]:
            def fac(kk=k, ww=w):
                return Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("sc", StandardScaler()),
                                 ("m", KNeighborsRegressor(n_neighbors=kk, weights=ww,
                                                            n_jobs=-1))])
            cvm, cvs, oof = cv_oof_filtered(fac, X_train, y_train, cv, winsor_filter(lo, hi))
            print(f"  Winsor[{lo:>4}-{hi}]+kNN k={k:>3} w={w:<8}: "
                  f"CV={cvm:+.5f}, OOF={oof:+.5f}")
            results.append({"name": f"Winsor[{lo}-{hi}]+kNN(k={k},{w})",
                            "cv_mean": cvm, "cv_std": cvs, "oof_r2": oof,
                            "factory": fac, "y_filter": winsor_filter(lo, hi)})


section("TOP 20 by OOF R²")
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("factory", "y_filter")}
                   for r in results]).sort_values("oof_r2", ascending=False).reset_index(drop=True)
print(df.head(20).to_string(index=False))


# Winner
best_name = df.iloc[0]["name"]
best = [r for r in results if r["name"] == best_name][0]
print(f"\n🏆 Winner: {best['name']}")
print(f"  CV  R² = {best['cv_mean']:+.5f} ± {best['cv_std']:.5f}")
print(f"  OOF R² = {best['oof_r2']:+.5f}")
print(f"  Expected LB ≈ {best['oof_r2'] + 0.011:+.4f}")
print(f"  Baseline target: +0.04599  →  gap remaining: "
      f"{0.04599 - (best['oof_r2'] + 0.011):+.4f}")


section("BUILD SUBMISSION")
pipe = best["factory"]()
X_fit, y_fit = best["y_filter"](X_train, y_train)
pipe.fit(X_fit, y_fit)
pred = pipe.predict(X_test)
print(f"Test pred stats: mean={pred.mean():.2f}, std={pred.std():.2f}, "
      f"min={pred.min():.2f}, max={pred.max():.2f}")

sub = pd.DataFrame({"Id": test_ids, "target": pred})
sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
short = (best["name"].lower().replace(" ", "").replace("=", "")
         .replace("(", "_").replace(")", "").replace(",", "_")
         .replace("α", "a").replace("[", "").replace("]", "")
         .replace("-", "to").replace(".", ""))
out = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_v07b_{short}.csv"
sub.to_csv(out, index=False)
print(f"\nSaved: {out}")

df.to_csv(ROOT / "docs" / "phase7b_results.csv", index=False)
