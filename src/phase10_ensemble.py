"""
Phase 10: Ensemble — Ridge(linear) + HGBR(tree) 가중 평균
============================================================
Phase 8 교훈: Winsor band sweep만으론 v7b 못 넘음 (sweet spot 박혀있음).
Phase 9 교훈: HGBR(tree) 단독으론 v7b 못 이김 (OOF +0.027 vs +0.033).

가설: Linear와 Tree는 서로 다른 종류 에러를 만듦.
      → 가중 평균 시 variance 감소로 둘 다 단독보다 좋아질 가능성.

실험:
  1. v7b (Ridge+winsor) 5-fold OOF + test prediction
  2. HGBR-sweet best 5-fold OOF + test prediction
  3. Multi-seed v7b (seed=0,42,100,2026) — variance 감소용
  4. Weight sweep: w_ridge in [0.0, 1.0] step 0.05
  5. Best ensemble → submission

발표 포인트: Linear(강의) + Tree(실전) 앙상블 = 각자 약점 보완.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]

KELLY_FU = 0.05051
PAWEL = 0.04877
V7B_LB = 0.04843


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def winsor_apply(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


def ridge_factory(alpha=1.0, seed=42):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=alpha, random_state=seed)),
    ])


def hgbr_sweet_factory(seed=42):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingRegressor(
            max_leaf_nodes=15, learning_rate=0.05, max_iter=300,
            random_state=seed)),
    ])


def cv_oof_and_test(fac, X, y, X_test, cv, winsor_fn=None):
    """Returns (oof_predictions, test_prediction averaged over folds)."""
    oof = np.zeros_like(y, dtype=float)
    test_preds = []
    for tr, va in cv.split(X):
        y_tr = winsor_fn(y[tr]) if winsor_fn else y[tr]
        pipe = fac().fit(X[tr], y_tr)
        oof[va] = pipe.predict(X[va])
        test_preds.append(pipe.predict(X_test))
    test_avg = np.mean(test_preds, axis=0)
    # final fit on full data for cleaner test pred
    y_full = winsor_fn(y) if winsor_fn else y
    full_pipe = fac().fit(X, y_full)
    test_full = full_pipe.predict(X_test)
    return oof, test_avg, test_full


# LOAD
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


# Component A: v7b (Ridge α=1 + Winsor[0.5-99.5])
section("Component A — v7b: Ridge α=1 + Winsor[0.5-99.5]")
oof_ridge, test_ridge_avg, test_ridge_full = cv_oof_and_test(
    lambda: ridge_factory(alpha=1.0, seed=42), X, y, X_test, cv,
    winsor_fn=lambda yy: winsor_apply(yy, 0.5, 99.5))
oof_r2_ridge = r2_score(y, oof_ridge)
print(f"  OOF R² = {oof_r2_ridge:+.5f}  (expected ≈ +0.0327)")


# Component B: HGBR-sweet best (leaf=15, lr=0.05, max_iter=300)
section("Component B — HGBR-sweet: leaf=15 lr=0.05 + Winsor[0.5-99.5]")
oof_hgbr, test_hgbr_avg, test_hgbr_full = cv_oof_and_test(
    hgbr_sweet_factory, X, y, X_test, cv,
    winsor_fn=lambda yy: winsor_apply(yy, 0.5, 99.5))
oof_r2_hgbr = r2_score(y, oof_hgbr)
print(f"  OOF R² = {oof_r2_hgbr:+.5f}  (expected ≈ +0.0274)")


# Component C: Multi-seed v7b (variance reduction)
section("Component C — Multi-seed Ridge (seed=0,42,100,2026)")
oof_multi_list, test_multi_list = [], []
for s in [0, 42, 100, 2026]:
    oof_s, _, test_s = cv_oof_and_test(
        lambda ss=s: ridge_factory(alpha=1.0, seed=ss), X, y, X_test,
        KFold(n_splits=N_FOLDS, shuffle=True, random_state=s),
        winsor_fn=lambda yy: winsor_apply(yy, 0.5, 99.5))
    oof_multi_list.append(oof_s)
    test_multi_list.append(test_s)
    print(f"  seed={s:>4}: OOF R² = {r2_score(y, oof_s):+.5f}")
oof_multi = np.mean(oof_multi_list, axis=0)
test_multi = np.mean(test_multi_list, axis=0)
print(f"  Multi-seed avg OOF R² = {r2_score(y, oof_multi):+.5f}")


# Correlation check
section("Component correlation analysis")
print(f"  corr(Ridge OOF, HGBR OOF)       = {np.corrcoef(oof_ridge, oof_hgbr)[0, 1]:.4f}")
print(f"  corr(Ridge OOF, Multi-seed OOF) = {np.corrcoef(oof_ridge, oof_multi)[0, 1]:.4f}")
print("  (낮은 상관 = 다양성 ↑ = 앙상블 이득 ↑)")


# Weight sweep — Ridge vs HGBR
section("Weight sweep: Ridge × w + HGBR × (1-w)")
best_w, best_r2 = 0.0, -1
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_ridge + (1 - w) * oof_hgbr
    r2 = r2_score(y, blend)
    marker = ""
    if r2 > best_r2:
        best_r2 = r2
        best_w = w
        marker = "  ← new best"
    print(f"  w_ridge={w:.2f}  OOF R² = {r2:+.5f}{marker}")
print(f"\n  🏆 Best weight: w_ridge={best_w:.2f}, OOF R² = {best_r2:+.5f}")
print(f"     Components alone: Ridge={oof_r2_ridge:+.5f}, HGBR={oof_r2_hgbr:+.5f}")
gain_vs_ridge = best_r2 - oof_r2_ridge
print(f"     Gain vs Ridge alone: {gain_vs_ridge:+.5f}")


# Weight sweep — Multi-seed vs HGBR
section("Weight sweep: Multi-seed Ridge × w + HGBR × (1-w)")
best_w2, best_r22 = 0.0, -1
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_multi + (1 - w) * oof_hgbr
    r2 = r2_score(y, blend)
    if r2 > best_r22:
        best_r22 = r2
        best_w2 = w
print(f"  🏆 Best: w_multi={best_w2:.2f}, OOF R² = {best_r22:+.5f}")
print(f"     Gain vs Multi-seed alone: {best_r22 - r2_score(y, oof_multi):+.5f}")


# BUILD SUBMISSIONS
section("BUILD CANDIDATE SUBMISSIONS")


def make_submission(test_pred, tag, oof_r2):
    sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    out = SUBMISSIONS_DIR / f"Baek_Seunghan_{tag}.csv"
    sub.to_csv(out, index=False)
    # Expected LB based on linear calibration (+0.016) — anchor v7b
    exp_lb_low = oof_r2 + 0.005  # tree-like
    exp_lb_high = oof_r2 + 0.016  # ridge-like
    print(f"\n  {tag}")
    print(f"     OOF R² = {oof_r2:+.5f}  →  Expected LB ≈ [{exp_lb_low:+.4f}, {exp_lb_high:+.4f}]")
    for label, ref in [("kelly_fu", KELLY_FU), ("Pawel", PAWEL), ("v7b", V7B_LB)]:
        status_high = "✅" if exp_lb_high > ref else "❌"
        print(f"     vs {label:9s} {ref}: high {status_high} ({exp_lb_high - ref:+.4f})")
    print(f"     → {out.name}")


# Candidate 1: Best Ridge+HGBR ensemble
test_ens1 = best_w * test_ridge_full + (1 - best_w) * test_hgbr_full
make_submission(test_ens1, f"v10a_ens_ridge{best_w:.2f}_hgbr{1-best_w:.2f}", best_r2)

# Candidate 2: Best Multi-seed Ridge + HGBR ensemble
test_ens2 = best_w2 * test_multi + (1 - best_w2) * test_hgbr_full
make_submission(test_ens2, f"v10b_ens_multiridge{best_w2:.2f}_hgbr{1-best_w2:.2f}", best_r22)

# Candidate 3: Multi-seed Ridge only (pure variance reduction)
make_submission(test_multi, "v10c_multiseed_ridge_only", r2_score(y, oof_multi))

print("\n✅ Phase 10 done.")
