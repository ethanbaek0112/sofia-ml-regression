"""
Phase 17: Day 1 Exploration — focused candidates for score push
=============================================================================
Goal: 0.0484 ceiling을 +0.0005~+0.0030 정도 nudging 시도.
원칙: Plain Ridge+Winsor 본질은 유지, 한 가지씩만 변경.

후보 categories (각자 다른 메커니즘):
  A. Micro Ridge α: 0.7/0.8/0.9/1.0/1.1/1.2/1.5 (현재 1.0)
  B. Micro Winsor band: [0.3-99.7]/[0.4-99.6]/[0.5-99.5]/[0.6-99.4]/[0.7-99.3]
  C. PCA(n_comp) + Ridge: 8, 10, 12, 14 components
  D. Sample-weighted Ridge: weight = 1/(1+|y|^p), p∈{0.3,0.5,1.0}
  E. Drop extreme rows: 학습 데이터에서 |y| > threshold sample 제거
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
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


def section(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

results = []


def fit_cv(make_pipe_fn, X_tr, y_tr, X_va, X_test):
    """Generic train + predict utility."""
    pipe = make_pipe_fn()
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_va), pipe.predict(X_test)


def standard_pipe(alpha=1.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=alpha, random_state=SEED)),
    ])


# ============================================================
section("v7b baseline")
oof_v7b = np.zeros_like(y, dtype=float)
test_v7b_folds = []
for tr, va in cv.split(X):
    pred_va, pred_test = fit_cv(lambda: standard_pipe(1.0), X[tr], winsor(y[tr]), X[va], X_test)
    oof_v7b[va] = pred_va
    test_v7b_folds.append(pred_test)
test_v7b = standard_pipe(1.0).fit(X, winsor(y)).predict(X_test)
r2_v7b = r2_score(y, oof_v7b)
print(f"v7b: OOF={r2_v7b:+.5f}  (LB 0.04843)")
results.append(("v17_baseline_v7b", r2_v7b, test_v7b))


# ============================================================
section("A — Micro Ridge α + Winsor[0.5-99.5]")
for alpha in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    oof = np.zeros_like(y, dtype=float)
    for tr, va in cv.split(X):
        p, _ = fit_cv(lambda a=alpha: standard_pipe(a), X[tr], winsor(y[tr]), X[va], X_test)
        oof[va] = p
    test_pred = standard_pipe(alpha).fit(X, winsor(y)).predict(X_test)
    r2 = r2_score(y, oof)
    tag = f"v17_A_alpha_{alpha}"
    delta = r2 - r2_v7b
    print(f"  α={alpha:<5}  OOF={r2:+.5f}  Δv7b={delta:+.5f}")
    results.append((tag, r2, test_pred))


# ============================================================
section("B — Micro Winsor band + Ridge α=1.0")
for lo, hi in [(0.3, 99.7), (0.4, 99.6), (0.5, 99.5), (0.6, 99.4), (0.7, 99.3), (0.8, 99.2), (1.0, 99.0)]:
    oof = np.zeros_like(y, dtype=float)
    for tr, va in cv.split(X):
        yt = winsor(y[tr], lo=lo, hi=hi)
        p, _ = fit_cv(lambda: standard_pipe(1.0), X[tr], yt, X[va], X_test)
        oof[va] = p
    test_pred = standard_pipe(1.0).fit(X, winsor(y, lo=lo, hi=hi)).predict(X_test)
    r2 = r2_score(y, oof)
    tag = f"v17_B_winsor_{lo}_{hi}"
    delta = r2 - r2_v7b
    print(f"  [{lo}-{hi}]  OOF={r2:+.5f}  Δv7b={delta:+.5f}")
    results.append((tag, r2, test_pred))


# ============================================================
section("C — PCA(n_comp) + Ridge")


def pca_pipe(n_comp, alpha=1.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=SEED)),
        ("m", Ridge(alpha=alpha, random_state=SEED)),
    ])


for n_comp in [5, 8, 10, 12, 14]:
    for alpha in [0.3, 1.0, 3.0]:
        oof = np.zeros_like(y, dtype=float)
        for tr, va in cv.split(X):
            p, _ = fit_cv(lambda nc=n_comp, a=alpha: pca_pipe(nc, a),
                          X[tr], winsor(y[tr]), X[va], X_test)
            oof[va] = p
        test_pred = pca_pipe(n_comp, alpha).fit(X, winsor(y)).predict(X_test)
        r2 = r2_score(y, oof)
        tag = f"v17_C_pca{n_comp}_a{alpha}"
        delta = r2 - r2_v7b
        print(f"  PCA n={n_comp:>2}, α={alpha}  OOF={r2:+.5f}  Δv7b={delta:+.5f}")
        results.append((tag, r2, test_pred))


# ============================================================
section("D — Sample-weighted Ridge")


def fit_weighted(X_tr, y_tr, X_va, X_test, alpha=1.0, weight_power=0.5):
    """Sample-weight = 1/(1+|y|^p) to down-weight extreme samples."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    X_tr_proc = pipe.fit_transform(X_tr)
    X_va_proc = pipe.transform(X_va)
    X_test_proc = pipe.transform(X_test)
    ridge = Ridge(alpha=alpha, random_state=SEED)
    weights = 1.0 / (1.0 + np.abs(y_tr) ** weight_power)
    ridge.fit(X_tr_proc, y_tr, sample_weight=weights)
    return ridge.predict(X_va_proc), ridge.predict(X_test_proc)


for power in [0.3, 0.5, 1.0]:
    for alpha in [0.3, 1.0, 3.0]:
        oof = np.zeros_like(y, dtype=float)
        for tr, va in cv.split(X):
            p, _ = fit_weighted(X[tr], winsor(y[tr]), X[va], X_test, alpha=alpha, weight_power=power)
            oof[va] = p
        # full fit
        _, test_pred = fit_weighted(X, winsor(y), X[:1], X_test, alpha=alpha, weight_power=power)
        r2 = r2_score(y, oof)
        tag = f"v17_D_swt_p{power}_a{alpha}"
        delta = r2 - r2_v7b
        print(f"  weight_power={power}, α={alpha}  OOF={r2:+.5f}  Δv7b={delta:+.5f}")
        results.append((tag, r2, test_pred))


# ============================================================
section("E — Drop extreme training rows (not just winsor)")
# Instead of clipping y, drop rows with |y| > threshold
for pct_keep in [99.0, 98.0, 95.0, 90.0]:
    threshold = np.percentile(np.abs(y), pct_keep)
    keep_mask = np.abs(y) <= threshold
    X_drop = X[keep_mask]
    y_drop = y[keep_mask]
    cv_drop = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # OOF on the original full y for fair comparison
    oof = np.zeros_like(y, dtype=float)
    # Train on dropped, predict on all
    for tr, va in cv.split(X):
        # Within each train fold, also drop extremes
        keep_tr = np.abs(y[tr]) <= threshold
        if keep_tr.sum() < 10:
            continue
        p = standard_pipe(1.0)
        p.fit(X[tr][keep_tr], y[tr][keep_tr])  # no winsor needed - we dropped
        oof[va] = p.predict(X[va])
    test_pred = standard_pipe(1.0).fit(X_drop, y_drop).predict(X_test)
    r2 = r2_score(y, oof)
    tag = f"v17_E_drop_keep{pct_keep}"
    delta = r2 - r2_v7b
    print(f"  keep {pct_keep}% (drop |y|>{threshold:.1f})  OOF={r2:+.5f}  Δv7b={delta:+.5f}")
    results.append((tag, r2, test_pred))


# ============================================================
section("🏆 TOP 15 — sorted by OOF (positive Δv7b only)")
results.sort(key=lambda r: -r[1])
shown = 0
for tag, r2, _ in results:
    delta = r2 - r2_v7b
    if shown < 15:
        marker = "🎯" if delta > 0 else "—"
        print(f"  {marker} {tag:<45s} OOF={r2:+.5f}  Δv7b={delta:+.5f}")
        shown += 1


section("📤 Save TOP 5 candidates (most promising)")
saved = set()
for tag, r2, tp in results[:5]:
    if tag in saved or "baseline" in tag:
        continue
    saved.add(tag)
    safe = tag.replace(".", "p")
    fname = f"Baek_Seunghan_{safe}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": tp})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    sub.to_csv(SUB_DIR / fname, index=False)
    print(f"  ✓ {fname:<55s} OOF={r2:+.5f}  Δv7b={r2 - r2_v7b:+.5f}")

print("\n✅ Phase 17 Day 1 exploration done.")
