"""
Phase 14c: Isotonic Calibration — safe alternative to ×2 rescaling
=============================================================================
Phase 14b 교훈: blind rescaling은 극단값 방향 틀리면 폭망 (LB -0.035).
Phase 14c 아이디어:
  - OOF (pred, y) pair로 monotone calibration function 학습
  - 방향(rank)은 보존, magnitude만 데이터 기반으로 조정
  - 폭망 risk 거의 없음 (identity 학습할 수도 있으니까)

추가:
  - Partial blending α: final = (1-α)*v7b + α*isotonic
  - α=0 이면 v7b (0.04843 보장)
  - α=1 이면 full isotonic
  - 중간값으로 점진적 위험 조절

여러 base prediction에 isotonic 적용:
  - v7b (Ridge plain)
  - v13 (Quantile + Ridge)
  - Multi-seed Ridge
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
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


def section(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


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


def make_ridge_pipe(alpha=1.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=alpha, random_state=SEED)),
    ])


def make_quantile_pipe(alpha=0.5, n_q=1000):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("trans", QuantileTransformer(n_quantiles=n_q, output_distribution="normal", random_state=SEED)),
        ("m", Ridge(alpha=alpha, random_state=SEED)),
    ])


def cv_oof_test(pipe_factory, X, y, X_test, cv):
    oof = np.zeros_like(y, dtype=float)
    test_fold_preds = []
    for tr, va in cv.split(X):
        pipe = pipe_factory()
        pipe.fit(X[tr], winsor(y[tr]))
        oof[va] = pipe.predict(X[va])
        test_fold_preds.append(pipe.predict(X_test))
    full = pipe_factory().fit(X, winsor(y))
    return oof, np.mean(test_fold_preds, axis=0), full.predict(X_test)


# ============================================================
section("Build base predictions")

print("\n[v7b — Ridge α=1]")
oof_v7b, _, test_v7b = cv_oof_test(make_ridge_pipe, X, y, X_test, cv)
r2_v7b = r2_score(y, oof_v7b)
print(f"  OOF = {r2_v7b:+.5f}  (LB known: 0.04843)")

print("\n[v13 — Quantile(n=1000, normal) + Ridge α=0.5]")
oof_v13, _, test_v13 = cv_oof_test(lambda: make_quantile_pipe(alpha=0.5, n_q=1000), X, y, X_test, cv)
r2_v13 = r2_score(y, oof_v13)
print(f"  OOF = {r2_v13:+.5f}  (LB known: 0.04820)")


# ============================================================
section("Isotonic Calibration — fit on OOF, apply to test")


def isotonic_calibrate(oof_pred, y_true, test_pred, out_min=None, out_max=None):
    """Fit isotonic on (oof_pred → y), apply to test_pred."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=out_min, y_max=out_max)
    ir.fit(oof_pred, y_true)
    oof_cal = ir.predict(oof_pred)
    test_cal = ir.predict(test_pred)
    return oof_cal, test_cal, ir


def cv_isotonic(base_pred_oof, y, base_pred_test, cv):
    """For honest OOF, fit isotonic inside CV folds too."""
    oof_cal = np.zeros_like(y, dtype=float)
    test_fold_preds = []
    for tr, va in cv.split(base_pred_oof):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(base_pred_oof[tr], y[tr])
        oof_cal[va] = ir.predict(base_pred_oof[va])
        test_fold_preds.append(ir.predict(base_pred_test))
    return oof_cal, np.mean(test_fold_preds, axis=0)


results = []

print("\n[Isotonic on v7b]")
# Naive: fit on full OOF, apply
oof_iso_v7b_naive, test_iso_v7b_naive, ir_v7b = isotonic_calibrate(oof_v7b, y, test_v7b)
print(f"  Naive (overfit OOF): OOF R² = {r2_score(y, oof_iso_v7b_naive):+.5f}  (optimistic — overfit)")
# Honest: CV-style isotonic
oof_iso_v7b_cv, test_iso_v7b_cv = cv_isotonic(oof_v7b, y, test_v7b, cv)
r2_iso_v7b_cv = r2_score(y, oof_iso_v7b_cv)
print(f"  CV (honest):          OOF R² = {r2_iso_v7b_cv:+.5f}  ← trust this")
results.append(("v14c_iso_v7b", r2_iso_v7b_cv, test_iso_v7b_naive))

print("\n[Isotonic on v13]")
oof_iso_v13_cv, test_iso_v13_cv = cv_isotonic(oof_v13, y, test_v13, cv)
r2_iso_v13_cv = r2_score(y, oof_iso_v13_cv)
oof_iso_v13_naive, test_iso_v13_naive, _ = isotonic_calibrate(oof_v13, y, test_v13)
print(f"  Naive: OOF R² = {r2_score(y, oof_iso_v13_naive):+.5f}")
print(f"  CV:    OOF R² = {r2_iso_v13_cv:+.5f}  ← trust")
results.append(("v14c_iso_v13", r2_iso_v13_cv, test_iso_v13_naive))


# ============================================================
section("Partial blending — (1-α)*v7b + α*iso_v7b")
print("\n  α=0 → pure v7b (safest, LB 0.04843 guaranteed)")
print("  α=1 → pure isotonic (full calibration)")
print()
print(f"  {'α':>5s}  {'OOF (CV)':>10s}  {'pred std':>10s}")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    blend_oof = (1 - alpha) * oof_v7b + alpha * oof_iso_v7b_cv
    blend_test = (1 - alpha) * test_v7b + alpha * test_iso_v7b_naive
    r2 = r2_score(y, blend_oof)
    tag = f"v14c_blend_iso_v7b_a{alpha:.2f}"
    print(f"  {alpha:.2f}    {r2:+.5f}     {blend_test.std():>10.2f}")
    results.append((tag, r2, blend_test))


# ============================================================
section("Isotonic on (v7b + v13 average) — diversity ensemble base")
oof_avg = 0.5 * oof_v7b + 0.5 * oof_v13
test_avg = 0.5 * test_v7b + 0.5 * test_v13
r2_avg = r2_score(y, oof_avg)
print(f"\n  Simple average OOF: {r2_avg:+.5f}")

oof_iso_avg_cv, test_iso_avg_cv = cv_isotonic(oof_avg, y, test_avg, cv)
oof_iso_avg_naive, test_iso_avg_naive, _ = isotonic_calibrate(oof_avg, y, test_avg)
r2_iso_avg_cv = r2_score(y, oof_iso_avg_cv)
print(f"  Iso-calibrated avg (CV): {r2_iso_avg_cv:+.5f}")
results.append(("v14c_iso_avg_v7b_v13", r2_iso_avg_cv, test_iso_avg_naive))

# Blend with v7b
print("\n  Blend: (1-α)*v7b + α*iso_avg:")
print(f"  {'α':>5s}  {'OOF (CV)':>10s}  {'pred std':>10s}")
for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
    bo = (1 - alpha) * oof_v7b + alpha * oof_iso_avg_cv
    bt = (1 - alpha) * test_v7b + alpha * test_iso_avg_naive
    r2 = r2_score(y, bo)
    tag = f"v14c_blend_iso_avg_a{alpha:.2f}"
    print(f"  {alpha:.2f}    {r2:+.5f}     {bt.std():>10.2f}")
    results.append((tag, r2, bt))


# ============================================================
section("🏆 TOP 10 — sorted by OOF (CV-honest)")
results.sort(key=lambda r: -r[1])
for tag, r2, _ in results[:10]:
    delta = r2 - r2_v7b
    marker = "🎯" if r2 > r2_v7b else "—"
    print(f"  {marker} {tag:<45s} OOF={r2:+.5f}  Δv7b={delta:+.5f}")


# ============================================================
section("📤 Save candidates — diverse risk levels")
# Save:
#   - Top by OOF (most aggressive)
#   - α=0.3 conservative blend
#   - α=0.5 medium blend
saved = set()


def save_one(tag, r2, test_pred):
    if tag in saved:
        return
    saved.add(tag)
    safe = tag.replace(".", "p")
    fname = f"Baek_Seunghan_{safe}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    out = SUB_DIR / fname
    sub.to_csv(out, index=False)
    print(f"  ✓ {fname:<60s} OOF={r2:+.5f}")


for tag, r2, tp in results[:3]:
    save_one(tag, r2, tp)
# Also explicitly save safer blends
for target_tag in ["v14c_blend_iso_v7b_a0.30", "v14c_blend_iso_v7b_a0.50"]:
    for tag, r2, tp in results:
        if tag == target_tag:
            save_one(tag, r2, tp)
            break

print("\n✅ Phase 14c done.")
