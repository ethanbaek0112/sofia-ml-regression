"""
Phase 11: Full Attack — Linear Family Sweep + Feature Engineering + Stacking
=============================================================================
Phase 10 교훈:
  - HGBR(tree) blend는 LB를 깎음 (calibration gap 무너짐).
  - Multi-seed Ridge는 LB에서 효과 없음 (모델이 이미 stable).
  - Sweet spot: Ridge + Winsor[0.5-99.5] = LB 0.04843 ceiling.

Phase 11 전략 (Tree 완전히 손 떼고 Linear 전공):
  Track A: Linear family — Ridge α grid, Huber, BayesianRidge, Lasso, ElasticNet
  Track B: Feature engineering — interactions + nonlinear transforms → Ridge
  Track C: Stacking — Ridge meta-learner on (Ridge OOF, HGBR OOF)
           HGBR 비중 자동 학습 (아마 0 근처 나올 듯)

평가 기준:
  - 1순위: OOF R² (5-fold KFold seed=42)
  - 2순위: Linear 계열은 v7b의 calibration gap (+0.016) 가정 OK
  - Tree 섞이면 gap 무너지므로 별도 표시
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
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
WINSOR_LO, WINSOR_HI = 0.5, 99.5

KELLY_FU = 0.05051
V7B_LB = 0.04843
V7B_OOF = 0.03266
LINEAR_CAL_GAP = 0.0157  # observed for v7b


def section(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def winsor(y, lo=WINSOR_LO, hi=WINSOR_HI):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


def make_pipe(model):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", model),
    ])


def cv_oof_test(model_factory, X, y, X_test, cv, winsor_fn=winsor):
    """5-fold OOF + test (averaged across folds + full-data refit)."""
    oof = np.zeros_like(y, dtype=float)
    test_fold_preds = []
    for tr, va in cv.split(X):
        y_tr = winsor_fn(y[tr]) if winsor_fn else y[tr]
        pipe = make_pipe(model_factory())
        pipe.fit(X[tr], y_tr)
        oof[va] = pipe.predict(X[va])
        test_fold_preds.append(pipe.predict(X_test))
    test_avg = np.mean(test_fold_preds, axis=0)
    y_full = winsor_fn(y) if winsor_fn else y
    full = make_pipe(model_factory()).fit(X, y_full)
    test_full = full.predict(X_test)
    return oof, test_avg, test_full


def report(name, oof_r2, is_linear=True):
    if is_linear:
        exp_lb = oof_r2 + LINEAR_CAL_GAP
        flag = "🏆" if exp_lb > KELLY_FU else "🥈" if exp_lb > V7B_LB else "—"
    else:
        exp_lb = oof_r2 + 0.005  # tree-like gap (conservative)
        flag = "⚠️"
    diff_v7b = oof_r2 - V7B_OOF
    print(f"  {flag} {name:<45s} OOF={oof_r2:+.5f}  (Δv7b={diff_v7b:+.5f})  exp_LB≈{exp_lb:+.4f}")
    return exp_lb


# ============================================================
# LOAD
# ============================================================
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"  Train: {X.shape}, Test: {X_test.shape}")
print(f"  v7b baseline: OOF={V7B_OOF:+.5f}, LB={V7B_LB:.5f}")

results = []  # list of (tag, oof_r2, test_pred, is_linear)


# ============================================================
# TRACK A: LINEAR FAMILY SWEEP
# ============================================================
section("TRACK A — Linear Family (all with Winsor[0.5-99.5])")

# A1. Ridge α grid (finer than before)
print("\n[A1] Ridge α grid")
ridge_results = []
for alpha in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    oof, _, t_full = cv_oof_test(
        lambda a=alpha: Ridge(alpha=a, random_state=SEED), X, y, X_test, cv)
    r2 = r2_score(y, oof)
    report(f"Ridge α={alpha}", r2)
    results.append((f"ridge_a{alpha}", r2, t_full, True))
    ridge_results.append((alpha, r2, t_full))

# A2. Huber Regression (built-in robustness, often w/o winsor)
print("\n[A2] Huber Regression (with + without winsor)")
for eps, alph, wf, tag in [
    (1.35, 0.0001, winsor, "huber_e1.35_a1e-4_winsor"),
    (1.35, 0.0001, None, "huber_e1.35_a1e-4_NOwinsor"),
    (1.5, 0.001, winsor, "huber_e1.5_a1e-3_winsor"),
    (2.0, 0.0001, winsor, "huber_e2.0_a1e-4_winsor"),
]:
    try:
        oof, _, t_full = cv_oof_test(
            lambda e=eps, a=alph: HuberRegressor(epsilon=e, alpha=a, max_iter=500),
            X, y, X_test, cv, winsor_fn=wf)
        r2 = r2_score(y, oof)
        report(tag, r2)
        results.append((tag, r2, t_full, True))
    except Exception as e:
        print(f"  ❌ {tag} failed: {e}")

# A3. Bayesian Ridge
print("\n[A3] Bayesian Ridge")
oof, _, t_full = cv_oof_test(
    lambda: BayesianRidge(), X, y, X_test, cv)
r2 = r2_score(y, oof)
report("bayesian_ridge_winsor", r2)
results.append(("bayesian_ridge_winsor", r2, t_full, True))

# A4. Lasso α grid
print("\n[A4] Lasso α grid")
for alpha in [0.001, 0.01, 0.1, 1.0]:
    oof, _, t_full = cv_oof_test(
        lambda a=alpha: Lasso(alpha=a, max_iter=5000, random_state=SEED),
        X, y, X_test, cv)
    r2 = r2_score(y, oof)
    report(f"lasso_a{alpha}", r2)
    results.append((f"lasso_a{alpha}", r2, t_full, True))


# ============================================================
# TRACK B: FEATURE ENGINEERING + RIDGE
# ============================================================
section("TRACK B — Feature Engineering → Ridge")

# Compute |correlations| to pick top features for interactions
corrs = []
for i, col in enumerate(FEATURE_COLS):
    c = np.corrcoef(X[:, i], y)[0, 1]
    corrs.append((col, abs(c), c))
corrs.sort(key=lambda r: -r[1])
print("\nTop-correlated features with target:")
for col, ac, c in corrs[:8]:
    print(f"  {col}: corr={c:+.4f} (|{ac:.4f}|)")

top_idx = [FEATURE_COLS.index(c[0]) for c in corrs[:8]]


def fe_nonlinear(Xm):
    """Add nonlinear univariate transforms for all 15 features."""
    parts = [Xm]
    parts.append(np.sign(Xm) * np.sqrt(np.abs(Xm)))  # signed sqrt
    parts.append(np.sign(Xm) * np.log1p(np.abs(Xm)))  # signed log
    parts.append(Xm ** 2)  # square (keep sign info via square is even)
    return np.hstack(parts)


def fe_interactions(Xm, idx_list):
    """Pairwise products of selected feature indices."""
    parts = [Xm]
    for i, j in combinations(idx_list, 2):
        parts.append((Xm[:, i] * Xm[:, j]).reshape(-1, 1))
    return np.hstack(parts)


def fe_full(Xm, idx_list):
    """Nonlinear + interactions among top features."""
    nl = fe_nonlinear(Xm)
    inter_cols = []
    for i, j in combinations(idx_list, 2):
        inter_cols.append((Xm[:, i] * Xm[:, j]).reshape(-1, 1))
    return np.hstack([nl] + inter_cols)


# B1. Nonlinear transforms only → Ridge
print("\n[B1] Nonlinear transforms (sqrt, log, square) → Ridge α grid")
X_nl = fe_nonlinear(X)
X_test_nl = fe_nonlinear(X_test)
for alpha in [1.0, 3.0, 10.0, 30.0, 100.0]:
    oof, _, t_full = cv_oof_test(
        lambda a=alpha: Ridge(alpha=a, random_state=SEED),
        X_nl, y, X_test_nl, cv)
    r2 = r2_score(y, oof)
    report(f"FE_nonlinear+ridge_a{alpha}", r2)
    results.append((f"fe_nl_ridge_a{alpha}", r2, t_full, True))

# B2. Top-K interactions only → Ridge
print("\n[B2] Top-K interactions only → Ridge")
for k in [4, 6, 8]:
    idx_k = top_idx[:k]
    X_inter = fe_interactions(X, idx_k)
    X_test_inter = fe_interactions(X_test, idx_k)
    for alpha in [1.0, 3.0, 10.0]:
        oof, _, t_full = cv_oof_test(
            lambda a=alpha: Ridge(alpha=a, random_state=SEED),
            X_inter, y, X_test_inter, cv)
        r2 = r2_score(y, oof)
        report(f"FE_inter_k{k}+ridge_a{alpha}", r2)
        results.append((f"fe_inter_k{k}_ridge_a{alpha}", r2, t_full, True))

# B3. Full FE (nonlinear + interactions) → Ridge
print("\n[B3] Full FE (nonlinear + top-K interactions) → Ridge")
for k in [4, 6, 8]:
    idx_k = top_idx[:k]
    X_full_fe = fe_full(X, idx_k)
    X_test_full_fe = fe_full(X_test, idx_k)
    for alpha in [3.0, 10.0, 30.0, 100.0]:
        oof, _, t_full = cv_oof_test(
            lambda a=alpha: Ridge(alpha=a, random_state=SEED),
            X_full_fe, y, X_test_full_fe, cv)
        r2 = r2_score(y, oof)
        report(f"FE_full_k{k}+ridge_a{alpha}", r2)
        results.append((f"fe_full_k{k}_ridge_a{alpha}", r2, t_full, True))


# ============================================================
# TRACK C: STACKING (Ridge meta on Ridge + HGBR OOFs)
# ============================================================
section("TRACK C — Stacking: Ridge meta-learner on (Ridge, HGBR) OOFs")

# Re-fit base models for clean stacking
print("\nFitting base models...")
oof_ridge, _, test_ridge_full = cv_oof_test(
    lambda: Ridge(alpha=1.0, random_state=SEED), X, y, X_test, cv)
print(f"  Base Ridge α=1: OOF={r2_score(y, oof_ridge):+.5f}")

oof_hgbr, _, test_hgbr_full = cv_oof_test(
    lambda: HistGradientBoostingRegressor(
        max_leaf_nodes=15, learning_rate=0.05, max_iter=300, random_state=SEED),
    X, y, X_test, cv)
print(f"  Base HGBR: OOF={r2_score(y, oof_hgbr):+.5f}")

# Stacking: train meta = LinearRegression on (oof_ridge, oof_hgbr)
meta_X = np.column_stack([oof_ridge, oof_hgbr])
meta_X_test = np.column_stack([test_ridge_full, test_hgbr_full])

# Use LinearRegression (no reg) and Ridge for meta
for meta_name, meta in [
    ("LinReg_meta", LinearRegression()),
    ("Ridge_a0.1_meta", Ridge(alpha=0.1)),
    ("Ridge_a1_meta", Ridge(alpha=1.0)),
]:
    # cv on meta features
    meta_oof = np.zeros_like(y, dtype=float)
    for tr, va in cv.split(meta_X):
        meta.fit(meta_X[tr], y[tr])  # don't winsor at meta level
        meta_oof[va] = meta.predict(meta_X[va])
    r2 = r2_score(y, meta_oof)
    # final fit
    meta.fit(meta_X, y)
    coefs = meta.coef_ if hasattr(meta, "coef_") else None
    test_pred = meta.predict(meta_X_test)
    coef_str = f"(coefs: ridge={coefs[0]:.3f}, hgbr={coefs[1]:.3f})" if coefs is not None else ""
    report(f"stack_{meta_name} {coef_str}", r2, is_linear=False)
    results.append((f"stack_{meta_name}", r2, test_pred, False))


# ============================================================
# FINAL: Top candidates summary
# ============================================================
section("🏆 TOP 10 CANDIDATES (sorted by OOF R²)")
results.sort(key=lambda r: -r[1])
for tag, r2, _, is_lin in results[:10]:
    exp_lb = r2 + (LINEAR_CAL_GAP if is_lin else 0.005)
    marker = "✅" if exp_lb > V7B_LB else "❌"
    kind = "L" if is_lin else "T"
    print(f"  {marker} [{kind}] {tag:<50s} OOF={r2:+.5f}  exp_LB≈{exp_lb:+.4f}")


# ============================================================
# BUILD SUBMISSIONS for top 3 LINEAR candidates
# ============================================================
section("📤 BUILD SUBMISSIONS — top 3 LINEAR candidates")
linear_results = [r for r in results if r[3]]
linear_results.sort(key=lambda r: -r[1])
for rank, (tag, r2, test_pred, _) in enumerate(linear_results[:3], 1):
    safe_tag = tag.replace(".", "p").replace("+", "_")
    fname = f"Baek_Seunghan_v11{chr(96 + rank)}_{safe_tag}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    out = SUB_DIR / fname
    sub.to_csv(out, index=False)
    exp_lb = r2 + LINEAR_CAL_GAP
    print(f"  rank{rank}: OOF={r2:+.5f}  exp_LB≈{exp_lb:+.4f}  →  {fname}")

print("\n✅ Phase 11 done.")
