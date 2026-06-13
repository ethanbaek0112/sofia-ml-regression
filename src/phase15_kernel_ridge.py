"""
Phase 15: KernelRidge (RBF) — last bullet for today
==================================================
RBF kernel Ridge: ridge in infinite-dim feature space.
- Naturally captures nonlinearity (different kind from trees)
- Same L2 regularization → predictions stay bounded
- Same feature count effectively → calibration gap might preserve
- NOT in lecture → maybe what engineers use
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]
SEED = 42


def winsor(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def cv_oof_test(model_factory, X, y, X_test, cv):
    oof = np.zeros_like(y, dtype=float)
    test_folds = []
    for tr, va in cv.split(X):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", model_factory()),
        ])
        pipe.fit(X[tr], winsor(y[tr]))
        oof[va] = pipe.predict(X[va])
        test_folds.append(pipe.predict(X_test))
    full = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", model_factory()),
    ]).fit(X, winsor(y))
    return oof, np.mean(test_folds, axis=0), full.predict(X_test)


print(f"\n{'=' * 76}\nKernelRidge RBF sweep — gamma × alpha\n{'=' * 76}")
# baseline v7b
oof_v7b, _, test_v7b = cv_oof_test(lambda: Ridge(alpha=1.0, random_state=SEED), X, y, X_test, cv)
r2_v7b = r2_score(y, oof_v7b)
print(f"v7b baseline:        OOF={r2_v7b:+.5f}  std={test_v7b.std():.2f}")
print()

results = []
print(f"  {'config':<35s} {'OOF':>10s}  {'Δv7b':>9s}  {'pred std':>10s}")
for gamma in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        oof, _, t_full = cv_oof_test(
            lambda g=gamma, a=alpha: KernelRidge(alpha=a, kernel="rbf", gamma=g),
            X, y, X_test, cv)
        r2 = r2_score(y, oof)
        delta = r2 - r2_v7b
        tag = f"v15_krr_rbf_g{gamma}_a{alpha}"
        marker = "🎯" if r2 > r2_v7b else "—"
        print(f"  {marker} krr_rbf γ={gamma:<6} α={alpha:<6}        {r2:+.5f}   {delta:+.5f}     {t_full.std():>10.2f}")
        results.append((tag, r2, t_full))

# Also try polynomial kernel
print()
print("Polynomial kernel sweep:")
for degree in [2, 3]:
    for alpha in [0.1, 1.0, 10.0]:
        oof, _, t_full = cv_oof_test(
            lambda d=degree, a=alpha: KernelRidge(alpha=a, kernel="polynomial", degree=d, coef0=1.0),
            X, y, X_test, cv)
        r2 = r2_score(y, oof)
        delta = r2 - r2_v7b
        tag = f"v15_krr_poly_d{degree}_a{alpha}"
        marker = "🎯" if r2 > r2_v7b else "—"
        print(f"  {marker} krr_poly d={degree} α={alpha:<6}             {r2:+.5f}   {delta:+.5f}     {t_full.std():>10.2f}")
        results.append((tag, r2, t_full))


print(f"\n{'=' * 76}\nTOP 10\n{'=' * 76}")
results.sort(key=lambda r: -r[1])
for tag, r2, _ in results[:10]:
    delta = r2 - r2_v7b
    marker = "🎯" if r2 > r2_v7b else "—"
    print(f"  {marker} {tag:<40s} OOF={r2:+.5f}  Δv7b={delta:+.5f}")


print(f"\n{'=' * 76}\nSave top 3\n{'=' * 76}")
for tag, r2, tp in results[:3]:
    safe = tag.replace(".", "p")
    fname = f"Baek_Seunghan_{safe}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": tp})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    sub.to_csv(SUB_DIR / fname, index=False)
    print(f"  ✓ {fname:<55s} OOF={r2:+.5f}")
print("\n✅ Phase 15 done.")
