"""
Phase 14d: Compression hypothesis + Target QuantileTransform
=============================================================================
지금까지 발견:
  - Train tail에 fit하는 모든 시도 → LB 폭망
  - Ridge+Winsor의 "압축"이 답이었음 (R² penalty bounded)

새 가설들:
  H1. 더 강한 압축 → 더 안전 → LB ↑?
      - v7b × {0.5, 0.7, 0.85, 1.0} 비교
      - 더 빡센 winsor band [2-98], [5-95]
  H2. Target QuantileTransform — distribution-matched prediction
      - target을 normal로 매핑 → Ridge → 역변환
      - exp 폭발 없음, 분포 정확히 매칭
  H3. 0-prediction baseline 확인 — sanity check
      - 0 예측 vs target 평균 예측 R² 비교
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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


def winsor_y(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


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


# Build v7b OOF and test
oof_v7b = np.zeros_like(y, dtype=float)
test_v7b_folds = []
for tr, va in cv.split(X):
    p = make_pipe(alpha=1.0).fit(X[tr], winsor_y(y[tr]))
    oof_v7b[va] = p.predict(X[va])
    test_v7b_folds.append(p.predict(X_test))
test_v7b = make_pipe(alpha=1.0).fit(X, winsor_y(y)).predict(X_test)
r2_v7b = r2_score(y, oof_v7b)
print(f"v7b base: OOF={r2_v7b:+.5f}  (LB known: 0.04843)")

results = []


# ============================================================
section("H1a — Sub-1 rescaling (MORE compression)")
print(f"\n  {'mult':<8s} {'OOF':>10s}  {'pred std':>10s}")
for mult in [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.10]:
    oof_re = oof_v7b * mult
    test_re = test_v7b * mult
    r2 = r2_score(y, oof_re)
    tag = f"v14d_compress_x{mult:.2f}"
    print(f"  ×{mult:.2f}    {r2:+.5f}      {test_re.std():>10.2f}")
    results.append((tag, r2, test_re))


# ============================================================
section("H1b — Stricter winsor band on target (more compression baked in)")
print(f"\n  {'band':<12s} {'α':<5s} {'OOF':>10s}  {'pred std':>10s}")
for lo, hi in [(0.5, 99.5), (1, 99), (2, 98), (5, 95), (10, 90), (25, 75)]:
    for alpha in [0.3, 1.0, 3.0]:
        oof = np.zeros_like(y, dtype=float)
        for tr, va in cv.split(X):
            yt = winsor_y(y[tr], lo=lo, hi=hi)
            p = make_pipe(alpha=alpha).fit(X[tr], yt)
            oof[va] = p.predict(X[va])
        yt_full = winsor_y(y, lo=lo, hi=hi)
        test_pred = make_pipe(alpha=alpha).fit(X, yt_full).predict(X_test)
        r2 = r2_score(y, oof)
        tag = f"v14d_winsor_{lo}_{hi}_a{alpha}"
        print(f"  [{lo}-{hi}]   {alpha}    {r2:+.5f}      {test_pred.std():>10.2f}")
        results.append((tag, r2, test_pred))


# ============================================================
section("H2 — Target QuantileTransform (predict normal-mapped y)")


def cv_qy(alpha=1.0, n_q=500, output_dist="normal"):
    oof = np.zeros_like(y, dtype=float)
    test_folds = []
    for tr, va in cv.split(X):
        qy = QuantileTransformer(n_quantiles=n_q, output_distribution=output_dist, random_state=SEED)
        y_tr_q = qy.fit_transform(y[tr].reshape(-1, 1)).ravel()
        p = make_pipe(alpha=alpha).fit(X[tr], y_tr_q)
        pred_q = p.predict(X[va])
        oof[va] = qy.inverse_transform(pred_q.reshape(-1, 1)).ravel()
        test_pred_q = p.predict(X_test)
        test_folds.append(qy.inverse_transform(test_pred_q.reshape(-1, 1)).ravel())
    # full
    qy_full = QuantileTransformer(n_quantiles=n_q, output_distribution=output_dist, random_state=SEED)
    y_full_q = qy_full.fit_transform(y.reshape(-1, 1)).ravel()
    p_full = make_pipe(alpha=alpha).fit(X, y_full_q)
    test_full_q = p_full.predict(X_test)
    test_full = qy_full.inverse_transform(test_full_q.reshape(-1, 1)).ravel()
    return oof, np.mean(test_folds, axis=0), test_full


print(f"\n  {'config':<35s} {'OOF':>10s}  {'pred std':>10s}")
for nq in [100, 500, 1000]:
    for alpha in [0.3, 1.0, 3.0, 10.0]:
        oof, _, t_full = cv_qy(alpha=alpha, n_q=nq, output_dist="normal")
        r2 = r2_score(y, oof)
        tag = f"v14d_qy_n{nq}_a{alpha}_normal"
        print(f"  qy_normal n={nq} α={alpha:<5}   {r2:+.5f}      {t_full.std():>10.2f}")
        results.append((tag, r2, t_full))


# ============================================================
section("H3 — Zero/Mean baseline R² sanity check")
zero_pred = np.zeros_like(y)
mean_pred = np.full_like(y, y.mean(), dtype=float)
median_pred = np.full_like(y, np.median(y), dtype=float)
print(f"  All-zero predictor:   R² = {r2_score(y, zero_pred):+.5f}")
print(f"  All-mean predictor:   R² = {r2_score(y, mean_pred):+.5f}  (by def = 0)")
print(f"  All-median predictor: R² = {r2_score(y, median_pred):+.5f}")


# ============================================================
section("🏆 TOP 15 — sorted by OOF (winsor + compression candidates)")
results.sort(key=lambda r: -r[1])
for tag, r2, _ in results[:15]:
    delta = r2 - r2_v7b
    marker = "🎯" if r2 > r2_v7b else "—"
    print(f"  {marker} {tag:<45s} OOF={r2:+.5f}  Δv7b={delta:+.5f}")


section("📤 Save diverse candidates")
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


# Top by OOF
for tag, r2, tp in results[:3]:
    save_one(tag, r2, tp)
# Save the best compression < 1 (different hypothesis)
for tag, r2, tp in results:
    if "compress_x0" in tag:
        save_one(tag, r2, tp)
        break
# Save best target-quantile
for tag, r2, tp in results:
    if "v14d_qy_" in tag:
        save_one(tag, r2, tp)
        break

print("\n✅ Phase 14d done.")
