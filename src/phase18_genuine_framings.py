"""
Phase 18 — Genuinely Different Framings Test
=============================================
Phase 17까지 결론: v7b family ceiling 0.0484, micro-tweak 다 무의미.
→ 진짜 다른 framing 5개를 OOF로 빠르게 시험.

A. Huber Regression           — robust loss (winsor 대용, principled)
B. X-feature winsorization    — input space에서 clip (지금까지 y만 했음)
C. Residual stacking           — v7b 후 residual 잡는 2nd model
D. TheilSen Regressor          — median-based robust
E. Rank-based target           — predict rank → inverse-transform

각 후보:
- OOF R² 계산
- v7b OOF와의 correlation (diversity 측정)
- 통과 기준: OOF >= 0.030 AND corr(v7b) < 0.97 (= 진짜 다른 정보)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge, TheilSenRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]

# Reference markers
KELLY = 0.05051  # actually higher now but use as ref
PAWEL = 0.04877
V7B_LB = 0.04843


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def winsor_y(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


def winsor_x(X, lo=1.0, hi=99.0):
    """Apply per-column winsorization to X."""
    X_out = X.copy().astype(float)
    for j in range(X.shape[1]):
        col = X_out[:, j]
        a, b = np.nanpercentile(col, [lo, hi])
        X_out[:, j] = np.clip(col, a, b)
    return X_out


def cv_oof_test(make_pipe, X, y, X_test, cv,
                winsor_y_fn=None, winsor_x_fn=None, target_transform=None,
                inverse_transform=None):
    """Generic 5-fold OOF + test prediction."""
    oof = np.zeros_like(y, dtype=float)
    test_preds = []
    for tr, va in cv.split(X):
        X_tr_raw, X_va_raw = X[tr], X[va]
        X_test_raw = X_test
        # X winsor (fit on train)
        if winsor_x_fn is not None:
            X_tr = winsor_x_fn(X_tr_raw)
            # apply to va & test using train percentiles
            pcts = np.array([np.nanpercentile(X_tr_raw[:, j], [1, 99])
                             for j in range(X_tr_raw.shape[1])])
            X_va = np.column_stack([np.clip(X_va_raw[:, j], pcts[j, 0], pcts[j, 1])
                                     for j in range(X_va_raw.shape[1])])
            X_te = np.column_stack([np.clip(X_test_raw[:, j], pcts[j, 0], pcts[j, 1])
                                     for j in range(X_test_raw.shape[1])])
        else:
            X_tr, X_va, X_te = X_tr_raw, X_va_raw, X_test_raw
        # y transform
        y_tr = y[tr]
        if winsor_y_fn is not None:
            y_tr = winsor_y_fn(y_tr)
        if target_transform is not None:
            y_tr_use = target_transform(y_tr)
        else:
            y_tr_use = y_tr
        pipe = make_pipe()
        pipe.fit(X_tr, y_tr_use)
        pred_va = pipe.predict(X_va)
        pred_te = pipe.predict(X_te)
        if inverse_transform is not None:
            pred_va = inverse_transform(pred_va, y_tr)
            pred_te = inverse_transform(pred_te, y_tr)
        oof[va] = pred_va
        test_preds.append(pred_te)
    return oof, np.mean(test_preds, axis=0)


# ───────────────────────────────────────────────────────────────────
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
print(f"train={X.shape}, test={X_test.shape}")


# ───────────────────────────────────────────────────────────────────
# Baseline: v7b for correlation reference
section("Baseline: v7b OOF (for diversity comparison)")
v7b_pipe = lambda: Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("m", Ridge(alpha=1.0, random_state=42)),
])
oof_v7b, test_v7b = cv_oof_test(
    v7b_pipe, X, y, X_test, cv,
    winsor_y_fn=lambda yy: winsor_y(yy, 0.5, 99.5))
oof_r2_v7b = r2_score(y, oof_v7b)
print(f"  v7b OOF R² = {oof_r2_v7b:+.5f}  (expected ≈ +0.0327)")


# ───────────────────────────────────────────────────────────────────
results = []  # (tag, oof_r2, corr_v7b, test_pred, note)

# A. Huber Regression
section("A. Huber Regression")
for eps in [1.0, 1.35, 2.0, 3.0]:
    make_pipe = lambda e=eps: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", HuberRegressor(epsilon=e, alpha=0.0001, max_iter=500)),
    ])
    try:
        oof, test_pred = cv_oof_test(make_pipe, X, y, X_test, cv,
                                     winsor_y_fn=lambda yy: winsor_y(yy, 0.5, 99.5))
        r2 = r2_score(y, oof)
        corr = np.corrcoef(oof, oof_v7b)[0, 1]
        print(f"  eps={eps:.2f}: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
        results.append((f"A_huber_eps{eps:.2f}", r2, corr, test_pred, "Huber"))
    except Exception as e:
        print(f"  eps={eps:.2f}: ERROR {e}")


# B. X-feature winsorization (with v7b underneath)
section("B. X-feature winsorization + v7b model")
for (xlo, xhi) in [(0.5, 99.5), (1.0, 99.0), (2.0, 98.0), (5.0, 95.0)]:
    oof, test_pred = cv_oof_test(
        v7b_pipe, X, y, X_test, cv,
        winsor_y_fn=lambda yy: winsor_y(yy, 0.5, 99.5),
        winsor_x_fn=lambda xx, lo=xlo, hi=xhi: winsor_x(xx, lo, hi))
    r2 = r2_score(y, oof)
    corr = np.corrcoef(oof, oof_v7b)[0, 1]
    print(f"  X-winsor [{xlo:.1f}-{xhi:.1f}]: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
    results.append((f"B_xwinsor_{xlo}_{xhi}", r2, corr, test_pred, "X-winsor"))


# C. Residual stacking (v7b → small Ridge on residuals)
section("C. Residual stacking — v7b → tiny Ridge on residuals")
# Step 1: get v7b OOF residuals (already have oof_v7b)
residuals = y - oof_v7b
print(f"  Residual stats: mean={residuals.mean():.2f}, std={residuals.std():.2f}")
# Step 2: train Ridge on residuals using same X via CV
for alpha2 in [10.0, 100.0, 1000.0]:
    res_pipe = lambda a=alpha2: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=a, random_state=42)),
    ])
    oof_res = np.zeros_like(y, dtype=float)
    test_res_list = []
    for tr, va in cv.split(X):
        pipe = res_pipe()
        # CRITICAL: train on residual from THIS fold's tr, but we have global oof_v7b
        # use clipped residual to avoid overfitting noise
        r_tr = np.clip(residuals[tr], np.percentile(residuals[tr], 1),
                       np.percentile(residuals[tr], 99))
        pipe.fit(X[tr], r_tr)
        oof_res[va] = pipe.predict(X[va])
        test_res_list.append(pipe.predict(X_test))
    test_res_avg = np.mean(test_res_list, axis=0)
    # Stacked prediction: v7b + residual
    oof_stack = oof_v7b + oof_res
    test_stack = test_v7b + test_res_avg
    r2 = r2_score(y, oof_stack)
    corr = np.corrcoef(oof_stack, oof_v7b)[0, 1]
    print(f"  α2={alpha2:>6}: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
    results.append((f"C_stack_a{alpha2:.0f}", r2, corr, test_stack, "Residual stack"))


# D. TheilSen Regression (median-based robust)
section("D. TheilSen Regression")
try:
    # TheilSen is SLOW — sample if needed
    make_pipe = lambda: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", TheilSenRegressor(random_state=42, max_subpopulation=2000,
                                  n_subsamples=200, n_jobs=-1)),
    ])
    oof, test_pred = cv_oof_test(make_pipe, X, y, X_test, cv,
                                  winsor_y_fn=lambda yy: winsor_y(yy, 0.5, 99.5))
    r2 = r2_score(y, oof)
    corr = np.corrcoef(oof, oof_v7b)[0, 1]
    print(f"  TheilSen: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
    results.append(("D_theilsen", r2, corr, test_pred, "TheilSen"))
except Exception as e:
    print(f"  TheilSen ERROR: {e}")


# E. Rank-based target (predict rank → inverse-transform)
section("E. Rank-based target")
# Train Ridge to predict y's rank, then inverse-transform back
def rank_transform(y_arr):
    return rankdata(y_arr) / len(y_arr)  # [0,1]


def inverse_rank(rank_pred, y_train_orig):
    """Given predicted rank in [0,1], look up the corresponding value
    from the training-y empirical distribution."""
    y_sorted = np.sort(y_train_orig)
    idx = np.clip((rank_pred * (len(y_sorted) - 1)).astype(int),
                  0, len(y_sorted) - 1)
    return y_sorted[idx]


rank_pipe = lambda: Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("m", Ridge(alpha=1.0, random_state=42)),
])
oof, test_pred = cv_oof_test(
    rank_pipe, X, y, X_test, cv,
    target_transform=rank_transform,
    inverse_transform=inverse_rank)
r2 = r2_score(y, oof)
corr = np.corrcoef(oof, oof_v7b)[0, 1]
print(f"  Rank-based: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
results.append(("E_rank", r2, corr, test_pred, "Rank target"))


# ───────────────────────────────────────────────────────────────────
# Summary table
section("SUMMARY — sorted by OOF R²")
results_sorted = sorted(results, key=lambda r: -r[1])
print(f"{'Tag':<28} {'OOF R²':>10} {'corr(v7b)':>11} {'Verdict':>20}")
print("─" * 72)
for tag, r2, corr, _, _ in results_sorted:
    # Verdict: promising if OOF high AND not too correlated
    if r2 < 0.025:
        verdict = "❌ weak OOF"
    elif corr > 0.99:
        verdict = "🟡 too similar to v7b"
    elif corr < 0.95 and r2 > 0.030:
        verdict = "✅ different info!"
    elif r2 > 0.034:
        verdict = "🟢 strong OOF"
    else:
        verdict = "🟡 modest"
    print(f"{tag:<28} {r2:>+10.5f} {corr:>11.4f}  {verdict:>20}")

# Save the most promising candidates only
section("BUILD CANDIDATE SUBMISSIONS (promising only)")
promising = [(tag, r2, corr, test_pred, note) for tag, r2, corr, test_pred, note in results
             if r2 > 0.030 and corr < 0.99]
if not promising:
    print("  ⚠️ No clearly promising candidate. v7b likely still best.")
else:
    for tag, r2, corr, test_pred, note in promising:
        sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
        sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
        out = SUBMISSIONS_DIR / f"Baek_Seunghan_v18_{tag}.csv"
        sub.to_csv(out, index=False)
        print(f"  ✅ {out.name}")
        print(f"     OOF={r2:+.5f}, corr(v7b)={corr:.4f}, note={note}")

print("\n✅ Phase 18 done.")
