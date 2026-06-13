"""
Phase 16: Safest last bullet — Median ensemble of LB-validated models + Quantile Reg check
=============================================================================
오늘 4번 시도 모두 v7b보다 떨어짐. 마지막 1슬롯은 안전하게.

후보:
  A. Median ensemble: median(v7b, v10c, v13)
     - 세 모델 모두 LB ~0.048 검증됨
     - Median = outlier robust → 셋이 다르게 틀린 sample들을 cleanup
     - 다 비슷한 prediction이라 worst case ≈ 0.048
  B. Quantile Regression (median, τ=0.5)
     - L1 loss = MSE보다 outlier robust
     - 강의 외 트릭, engineer 즐겨씀
     - 미지수 — OOF 보고 결정
  C. Mean ensemble of (A) — 비교용
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]
SEED = 42


def winsor(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


def section(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def cv_oof_test(pipe_factory):
    oof = np.zeros_like(y, dtype=float)
    test_folds = []
    for tr, va in cv.split(X):
        p = pipe_factory()
        p.fit(X[tr], winsor(y[tr]))
        oof[va] = p.predict(X[va])
        test_folds.append(p.predict(X_test))
    full = pipe_factory().fit(X, winsor(y))
    return oof, full.predict(X_test)


# ============================================================
section("Build LB-validated base predictions")

# v7b: Ridge α=1 + winsor
oof_v7b, test_v7b = cv_oof_test(lambda: Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("m", Ridge(alpha=1.0, random_state=SEED)),
]))
print(f"v7b  OOF={r2_score(y, oof_v7b):+.5f}  (LB 0.04843)")

# v10c: Multi-seed Ridge (avg of 4)
oof_multi_list, test_multi_list = [], []
for s in [0, 42, 100, 2026]:
    cv_s = KFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
    oof_s = np.zeros_like(y, dtype=float)
    test_folds = []
    for tr, va in cv_s.split(X):
        p = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=1.0, random_state=s)),
        ]).fit(X[tr], winsor(y[tr]))
        oof_s[va] = p.predict(X[va])
        test_folds.append(p.predict(X_test))
    full = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=1.0, random_state=s)),
    ]).fit(X, winsor(y))
    oof_multi_list.append(oof_s)
    test_multi_list.append(full.predict(X_test))
oof_v10c = np.mean(oof_multi_list, axis=0)
test_v10c = np.mean(test_multi_list, axis=0)
print(f"v10c OOF={r2_score(y, oof_v10c):+.5f}  (LB 0.04843)")

# v13: Quantile(n=1000, normal) + Ridge α=0.5
oof_v13, test_v13 = cv_oof_test(lambda: Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("trans", QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)),
    ("m", Ridge(alpha=0.5, random_state=SEED)),
]))
print(f"v13  OOF={r2_score(y, oof_v13):+.5f}  (LB 0.04820)")


# ============================================================
section("Ensembles of LB-validated bases")
oof_stack = np.column_stack([oof_v7b, oof_v10c, oof_v13])
test_stack = np.column_stack([test_v7b, test_v10c, test_v13])

# Correlations
print("\n  Correlations (high = similar, low = diverse):")
labels = ["v7b", "v10c", "v13"]
for i in range(3):
    for j in range(i + 1, 3):
        c = np.corrcoef(oof_stack[:, i], oof_stack[:, j])[0, 1]
        print(f"    {labels[i]:>5s} vs {labels[j]:<5s}: {c:+.4f}")

candidates = []

# Mean ensemble
oof_mean = oof_stack.mean(axis=1)
test_mean = test_stack.mean(axis=1)
r2_mean = r2_score(y, oof_mean)
print(f"\n  MEAN(v7b, v10c, v13):     OOF={r2_mean:+.5f}  std={test_mean.std():.2f}")
candidates.append(("v16_mean_v7b_v10c_v13", r2_mean, test_mean))

# Median ensemble
oof_med = np.median(oof_stack, axis=1)
test_med = np.median(test_stack, axis=1)
r2_med = r2_score(y, oof_med)
print(f"  MEDIAN(v7b, v10c, v13):   OOF={r2_med:+.5f}  std={test_med.std():.2f}")
candidates.append(("v16_median_v7b_v10c_v13", r2_med, test_med))

# Weighted: favor v10c slightly (best OOF)
oof_w = 0.30 * oof_v7b + 0.40 * oof_v10c + 0.30 * oof_v13
test_w = 0.30 * test_v7b + 0.40 * test_v10c + 0.30 * test_v13
r2_w = r2_score(y, oof_w)
print(f"  WEIGHTED 0.3/0.4/0.3:     OOF={r2_w:+.5f}  std={test_w.std():.2f}")
candidates.append(("v16_weighted_v7b30_v10c40_v13_30", r2_w, test_w))


# ============================================================
section("Quantile Regression check (median, τ=0.5)")
print("\n  Trying QuantileRegressor (might be slow on 2500 samples)...")
print(f"  {'config':<30s} {'OOF':>10s}  {'pred std':>10s}")
for alpha_q in [0.0001, 0.001, 0.01]:
    try:
        oof_q = np.zeros_like(y, dtype=float)
        test_folds_q = []
        for tr, va in cv.split(X):
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", QuantileRegressor(quantile=0.5, alpha=alpha_q, solver="highs")),
            ])
            pipe.fit(X[tr], winsor(y[tr]))
            oof_q[va] = pipe.predict(X[va])
            test_folds_q.append(pipe.predict(X_test))
        full = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", QuantileRegressor(quantile=0.5, alpha=alpha_q, solver="highs")),
        ]).fit(X, winsor(y))
        test_q = full.predict(X_test)
        r2_q = r2_score(y, oof_q)
        print(f"  QuantileReg τ=0.5 α={alpha_q}        {r2_q:+.5f}     {test_q.std():>10.2f}")
        candidates.append((f"v16_qreg_a{alpha_q}", r2_q, test_q))
    except Exception as e:
        print(f"  QuantileReg α={alpha_q} failed: {e}")


# ============================================================
section("🏆 Ranking — by OOF + safety analysis")
candidates.sort(key=lambda r: -r[1])
r2_v7b_actual = r2_score(y, oof_v7b)
for tag, r2, tp in candidates:
    delta = r2 - r2_v7b_actual
    print(f"  {tag:<40s} OOF={r2:+.5f}  Δv7b={delta:+.5f}")


section("📤 Save")
saved = set()
for tag, r2, tp in candidates[:3]:
    if tag in saved:
        continue
    saved.add(tag)
    safe = tag.replace(".", "p")
    fname = f"Baek_Seunghan_{safe}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": tp})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    sub.to_csv(SUB_DIR / fname, index=False)
    print(f"  ✓ {fname:<55s} OOF={r2:+.5f}")
# Always save median ensemble explicitly
for tag, r2, tp in candidates:
    if "median" in tag and tag not in saved:
        saved.add(tag)
        safe = tag.replace(".", "p")
        fname = f"Baek_Seunghan_{safe}.csv"
        sub = pd.DataFrame({"Id": test["Id"].values, "target": tp})
        sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
        sub.to_csv(SUB_DIR / fname, index=False)
        print(f"  ✓ {fname:<55s} OOF={r2:+.5f}  (median safety)")
        break

print("\n✅ Phase 16 done.")
