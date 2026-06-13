"""
Phase 12: Ablation + Blend — gap 회복 시도
=============================================================================
Phase 11 충격: FE_nonlinear → OOF +0.0083 향상했는데 LB는 +0.0009만 반영.
   계산: gap이 OOF 향상에 거의 1:1로 깎임 (gap = 0.016 → 0.001).

Phase 12 가설:
  1. nonlinear FE 3종(sqrt, log, square) 중 일부만 gap을 깎고 일부는 OK일 수 있음
     → 각각 따로 + 둘만 조합으로 분리 (ablation)
  2. v7b(gap 좋음) + v11a(OOF 좋음) blend로 gap을 부분 회복 가능
     → 다양한 weight sweep
  3. 극도 regularization (α=50~500) → FE 정보 일부만 살릴 때 gap 회복?

평가:
  - OOF만으론 못 믿음 (Phase 11 교훈)
  - 따라서 LB 추정은 "v7b-style gap"이 회복 가능한지 보수적으로 평가
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
SUB_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]
SEED = 42
WINSOR_LO, WINSOR_HI = 0.5, 99.5

V7B_LB = 0.04843
V11A_LB = 0.04190
V11A_OOF = 0.0410
V7B_OOF = 0.03266


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
    oof = np.zeros_like(y, dtype=float)
    test_fold_preds = []
    for tr, va in cv.split(X):
        y_tr = winsor_fn(y[tr]) if winsor_fn else y[tr]
        pipe = make_pipe(model_factory())
        pipe.fit(X[tr], y_tr)
        oof[va] = pipe.predict(X[va])
        test_fold_preds.append(pipe.predict(X_test))
    y_full = winsor_fn(y) if winsor_fn else y
    full = make_pipe(model_factory()).fit(X, y_full)
    test_full = full.predict(X_test)
    return oof, np.mean(test_fold_preds, axis=0), test_full


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


# ============================================================
# Step 1: Build base predictions
# ============================================================
section("STEP 1 — Base predictions (v7b plain Ridge, v11a nonlinear FE)")

# v7b: plain Ridge α=1
oof_v7b, _, test_v7b = cv_oof_test(
    lambda: Ridge(alpha=1.0, random_state=SEED), X, y, X_test, cv)
print(f"  v7b OOF = {r2_score(y, oof_v7b):+.5f}  (LB known: {V7B_LB})")


def t_sqrt(Xm):
    return np.sign(Xm) * np.sqrt(np.abs(Xm))


def t_log(Xm):
    return np.sign(Xm) * np.log1p(np.abs(Xm))


def t_square(Xm):
    return Xm ** 2


# ============================================================
# Step 2: ABLATION — single transform variants
# ============================================================
section("STEP 2 — ABLATION: Which transform is the LB killer?")

ablations = {
    "sqrt_only":   (lambda Xm: np.hstack([Xm, t_sqrt(Xm)]),                "Xm + sgn·√|x|"),
    "log_only":    (lambda Xm: np.hstack([Xm, t_log(Xm)]),                 "Xm + sgn·log(1+|x|)"),
    "square_only": (lambda Xm: np.hstack([Xm, t_square(Xm)]),              "Xm + x²"),
    "sqrt_log":    (lambda Xm: np.hstack([Xm, t_sqrt(Xm), t_log(Xm)]),    "Xm + sqrt + log"),
    "sqrt_sq":     (lambda Xm: np.hstack([Xm, t_sqrt(Xm), t_square(Xm)]), "Xm + sqrt + square"),
    "log_sq":      (lambda Xm: np.hstack([Xm, t_log(Xm), t_square(Xm)]),  "Xm + log + square"),
    "all_three":   (lambda Xm: np.hstack([Xm, t_sqrt(Xm), t_log(Xm), t_square(Xm)]), "Xm + sqrt + log + square (=v11a)"),
}

# For each transform set, sweep regularization
ablation_results = {}  # tag -> (best_alpha, best_oof, best_test_pred)
print(f"\n{'transform set':<14s} {'desc':<35s} {'α':>5s}  {'OOF':>10s}")
for tag, (fe_fn, desc) in ablations.items():
    Xfe = fe_fn(X)
    Xfe_test = fe_fn(X_test)
    best = (-1, None, None)  # oof_r2, alpha, test_pred
    sweep = []
    for alpha in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        oof, _, t_full = cv_oof_test(
            lambda a=alpha: Ridge(alpha=a, random_state=SEED), Xfe, y, Xfe_test, cv)
        r2 = r2_score(y, oof)
        sweep.append((alpha, r2))
        if r2 > best[0]:
            best = (r2, alpha, t_full)
    print(f"  {tag:<14s} {desc:<35s} {best[1]:>5g}  {best[0]:+.5f}")
    for alpha, r2 in sweep:
        print(f"    α={alpha:>5g}  OOF={r2:+.5f}")
    ablation_results[tag] = (best[1], best[0], best[2])


# ============================================================
# Step 3: BLEND v7b + v11a (gap recovery)
# ============================================================
section("STEP 3 — BLEND: v7b (gap+0.016) + v11a (OOF best)")

# rebuild v11a (all three) with α=1
Xfe11 = np.hstack([X, t_sqrt(X), t_log(X), t_square(X)])
Xfe11_test = np.hstack([X_test, t_sqrt(X_test), t_log(X_test), t_square(X_test)])
oof_v11a, _, test_v11a = cv_oof_test(
    lambda: Ridge(alpha=1.0, random_state=SEED), Xfe11, y, Xfe11_test, cv)
print(f"  v11a OOF = {r2_score(y, oof_v11a):+.5f}  (LB known: {V11A_LB})")

# Calibration anchor: v7b_LB = 0.04843, v11a_LB = 0.04190
# If predictions on test scale linearly, then blend LB ≈ w*v7b_LB + (1-w)*v11a_LB
# But we can also estimate via OOF blend
print("\n  weight sweep (w=v7b weight):")
print(f"  {'w':>5s}  {'OOF':>10s}  {'lin_est_LB':>11s}")
blend_oofs = {}
for w in np.arange(0.0, 1.01, 0.1):
    blend_oof = w * oof_v7b + (1 - w) * oof_v11a
    r2 = r2_score(y, blend_oof)
    lin_lb = w * V7B_LB + (1 - w) * V11A_LB
    blend_oofs[round(w, 2)] = (r2, lin_lb, w * test_v7b + (1 - w) * test_v11a)
    print(f"  {w:.2f}   {r2:+.5f}   {lin_lb:.5f}")

# Pick a few interesting blends
# Logic: heavy v7b weight should recover gap; small v11a contribution may slightly add signal
selected_blends = [0.5, 0.7, 0.8]


# ============================================================
# Step 4: Pure v7b refresh (control)
# ============================================================
section("STEP 4 — Build candidate submissions")

candidates = []

# Best ablation per transform
print("\n[Ablation winners]")
for tag, (alpha, r2, test_pred) in ablation_results.items():
    candidates.append((f"v12_abl_{tag}_a{alpha:g}", r2, test_pred, "ablation"))
    print(f"  v12_abl_{tag}_a{alpha:g}  OOF={r2:+.5f}")

# Blends
print("\n[Blends]")
for w in selected_blends:
    r2, lin_lb, test_pred = blend_oofs[w]
    tag = f"v12_blend_v7b{w:.2f}_v11a{1-w:.2f}"
    candidates.append((tag, r2, test_pred, f"blend lin_lb={lin_lb:.4f}"))
    print(f"  {tag}  OOF={r2:+.5f}  lin_est_LB={lin_lb:.4f}")


# ============================================================
# Step 5: pick TOP 3 — by 두 가지 기준
# ============================================================
section("🏆 RANKING — by OOF (top 5)")
candidates_sorted = sorted(candidates, key=lambda c: -c[1])
for i, (tag, r2, _, note) in enumerate(candidates_sorted[:5], 1):
    print(f"  {i}. {tag:<45s} OOF={r2:+.5f}  [{note}]")


# Save TOP 3 by OOF + the most conservative blend (w=0.7 v7b)
section("📤 SAVE CANDIDATES")
to_save = []
for tag, r2, test_pred, note in candidates_sorted[:3]:
    to_save.append((tag, r2, test_pred, note))
# Also force-include blend w=0.7 if not already
blend_07_tag = "v12_blend_v7b0.70_v11a0.30"
if not any(blend_07_tag == c[0] for c in to_save):
    for c in candidates:
        if c[0] == blend_07_tag:
            to_save.append(c)
            break

for tag, r2, test_pred, note in to_save:
    safe_tag = tag.replace(".", "p")
    fname = f"Baek_Seunghan_{safe_tag}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    out = SUB_DIR / fname
    sub.to_csv(out, index=False)
    print(f"  ✓ {fname:<60s} OOF={r2:+.5f}  [{note}]")

print("\n✅ Phase 12 done.")
