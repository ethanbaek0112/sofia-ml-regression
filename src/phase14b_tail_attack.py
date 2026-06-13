"""
Phase 14b: Tail attack — log-transform target + rescaling experiments
=============================================================================
진단 발견:
  - Target skew=+13.58, kurtosis=+706 (극단적 heavy tail)
  - Ridge+Winsor prediction은 9.3x 압축됨
  - 우리 LB가 0.048에 갇힌 이유: tail 예측 불가

전략:
  A. Signed log-transform target: y' = sign(y) * log(1 + |y|)
     → 학습 후 역변환. skewness 해결하면서 tail 유지
  B. Partial rescaling: v7b prediction × {1.5, 2, 3, 5, 9.3}
     → 압축 해제 정도 조절
  C. log + winsor combined: log 변환 후 가벼운 winsor
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


def section(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def winsor(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


def signed_log(y):
    return np.sign(y) * np.log1p(np.abs(y))


def inv_signed_log(z):
    return np.sign(z) * (np.expm1(np.abs(z)))


# LOAD
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def make_pipe(alpha=1.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=alpha, random_state=SEED)),
    ])


def cv_oof_test(y_transform, y_inv_transform, alpha=1.0,
                winsor_z=None, X=X, y=y, X_test=X_test, cv=cv):
    """Train on transformed y, evaluate in original y space."""
    oof_orig = np.zeros_like(y, dtype=float)
    test_fold_preds = []
    for tr, va in cv.split(X):
        y_tr_t = y_transform(y[tr])
        if winsor_z is not None:
            y_tr_t = np.clip(y_tr_t, *np.percentile(y_tr_t, [winsor_z, 100 - winsor_z]))
        pipe = make_pipe(alpha=alpha).fit(X[tr], y_tr_t)
        z_va = pipe.predict(X[va])
        oof_orig[va] = y_inv_transform(z_va)
        test_fold_preds.append(y_inv_transform(pipe.predict(X_test)))
    # Full refit
    y_t = y_transform(y)
    if winsor_z is not None:
        y_t = np.clip(y_t, *np.percentile(y_t, [winsor_z, 100 - winsor_z]))
    full = make_pipe(alpha=alpha).fit(X, y_t)
    test_full = y_inv_transform(full.predict(X_test))
    return oof_orig, np.mean(test_fold_preds, axis=0), test_full


# ============================================================
section("A — Signed log transform target")
print(f"\n  Target stats: mean={y.mean():.2f}, std={y.std():.2f}, skew~13.6")
print(f"  Signed-log target: mean={signed_log(y).mean():.4f}, std={signed_log(y).std():.4f}")

results = []
print(f"\n  {'config':<45s} {'OOF R²':>10s}")
# A1. Signed log + no winsor on z
for alpha in [0.3, 1.0, 3.0, 10.0]:
    oof, _, t_full = cv_oof_test(signed_log, inv_signed_log, alpha=alpha)
    r2 = r2_score(y, oof)
    print(f"  signed_log + Ridge α={alpha} (no z-winsor)              {r2:+.5f}")
    results.append((f"v14b_slog_a{alpha}", r2, t_full))

# A2. Signed log + winsor in z-space
for alpha in [0.3, 1.0, 3.0]:
    for wz in [0.5, 1.0, 2.0]:
        oof, _, t_full = cv_oof_test(signed_log, inv_signed_log, alpha=alpha, winsor_z=wz)
        r2 = r2_score(y, oof)
        print(f"  signed_log + Ridge α={alpha} + z-winsor[{wz}-{100-wz}] {r2:+.5f}")
        results.append((f"v14b_slog_a{alpha}_wz{wz}", r2, t_full))


# ============================================================
section("B — Partial rescaling v7b prediction (post-hoc)")
# Build v7b pred
full_v7b = make_pipe(alpha=1.0).fit(X, winsor(y))
test_v7b = full_v7b.predict(X_test)

oof_v7b = np.zeros_like(y, dtype=float)
for tr, va in cv.split(X):
    p = make_pipe(alpha=1.0).fit(X[tr], winsor(y[tr]))
    oof_v7b[va] = p.predict(X[va])

base_std = test_v7b.std()
target_std = y.std()
full_ratio = target_std / base_std
print(f"\n  v7b test pred std: {base_std:.2f},  target std: {target_std:.2f},  full ratio: {full_ratio:.3f}")

print(f"\n  {'multiplier':<15s} {'OOF R²':>10s}  {'pred std':>10s}")
for mult in [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, full_ratio]:
    oof_re = oof_v7b * mult
    test_re = test_v7b * mult
    r2 = r2_score(y, oof_re)
    tag = f"v14b_rescale_x{mult:.2f}"
    print(f"  ×{mult:.3f}                  {r2:+.5f}     {test_re.std():>10.2f}")
    results.append((tag, r2, test_re))


# ============================================================
section("C — Signed log + post-rescaling")
# Best signed log alpha
best_slog = sorted([r for r in results if r[0].startswith("v14b_slog_a") and "wz" not in r[0]],
                   key=lambda r: -r[1])[0]
print(f"\n  Best signed log so far: {best_slog[0]} OOF={best_slog[1]:+.5f}")

# Get pred for best slog
best_alpha = float(best_slog[0].split("_a")[1])
oof_slog, _, test_slog = cv_oof_test(signed_log, inv_signed_log, alpha=best_alpha)
slog_std = test_slog.std()
print(f"  Test slog pred std: {slog_std:.2f}")

# Try mild rescaling on top
for mult in [1.0, 1.25, 1.5, 2.0]:
    oof_re = oof_slog * mult
    test_re = test_slog * mult
    r2 = r2_score(y, oof_re)
    tag = f"v14b_slog_a{best_alpha}_x{mult:.2f}"
    print(f"  signed_log + ×{mult}: OOF={r2:+.5f}, pred_std={test_re.std():.2f}")
    results.append((tag, r2, test_re))


# ============================================================
section("🏆 TOP 10 — by OOF R²")
results.sort(key=lambda r: -r[1])
for tag, r2, _ in results[:10]:
    marker = "🎯" if r2 > 0.033 else "—"
    print(f"  {marker} {tag:<50s} OOF={r2:+.5f}")


section("📤 SAVE diverse candidates")
# Save:
#   1. Best by OOF
#   2. Best signed log (no rescale) — different paradigm
#   3. Best rescale  — different paradigm
#   4. Conservative slog+small rescale
# Try to ensure diversity
saved = set()


def save_one(tag, r2, test_pred):
    if tag in saved:
        return
    saved.add(tag)
    safe = tag.replace(".", "p").replace("-", "m")
    fname = f"Baek_Seunghan_{safe}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    out = SUB_DIR / fname
    sub.to_csv(out, index=False)
    print(f"  ✓ {fname:<60s} OOF={r2:+.5f}")


# Top OOF
for tag, r2, tp in results[:3]:
    save_one(tag, r2, tp)
# Best slog only (paradigm diversity)
for tag, r2, tp in results:
    if tag.startswith("v14b_slog_a") and "x" not in tag and "wz" not in tag:
        save_one(tag, r2, tp)
        break
# Best rescale
for tag, r2, tp in results:
    if tag.startswith("v14b_rescale_x") and "x1.00" not in tag:
        save_one(tag, r2, tp)
        break

print("\n✅ Phase 14b done.")
