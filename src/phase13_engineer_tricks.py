"""
Phase 13: Engineer Tricks — Adversarial Val + Quantile + Bootstrap Ridge
=============================================================================
배경: 상위 4명(Manish, Kelly, Pawel, Chiikawa)이 모두 0.0496+ 달성.
      강의 외 트릭을 쓰는 게 확실 (engineer pros).

Phase 11/12 교훈:
  - OOF↑가 LB↑로 안 이어짐 → CV와 LB가 다른 분포일 가능성
  - Winsor + Ridge가 답인데 더 강한 robust 변환이 답일 수도

Phase 13 가설 (Tier 1 트릭):
  T1. Adversarial Validation:
      - RF가 train vs test 구분 가능 → covariate shift 존재
      - shift 큰 feature는 가중치 줄이거나 변환
  T2. QuantileTransformer:
      - 각 feature를 균등분포로 매핑 (outlier 완전 무력화)
      - 분포 shift에도 robust
  T3. Bootstrap Ridge ensemble (median):
      - 200 bootstraps × Ridge, median aggregation
      - multi-seed (mean)보다 robust
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]
SEED = 42
WINSOR_LO, WINSOR_HI = 0.5, 99.5

V7B_LB = 0.04843


def section(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def winsor(y, lo=WINSOR_LO, hi=WINSOR_HI):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


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
# T1: ADVERSARIAL VALIDATION
# ============================================================
section("T1 — ADVERSARIAL VALIDATION: train vs test distinguishable?")

# Impute first (NaN handling)
imp = SimpleImputer(strategy="median")
X_all_imp = imp.fit_transform(np.vstack([X, X_test]))
X_imp = X_all_imp[:len(X)]
X_test_imp = X_all_imp[len(X):]

# Stack with labels
X_adv = np.vstack([X_imp, X_test_imp])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(X_test))])

# RF classifier — can it distinguish?
clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED, n_jobs=-1)
scores = cross_val_score(clf, X_adv, y_adv, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"  RF train-vs-test AUC = {scores.mean():.4f} ± {scores.std():.4f}")
print(f"  (0.50 = 분포 동일, 0.60+ = shift 존재, 0.70+ = 심각)")

# Feature importances
clf.fit(X_adv, y_adv)
importances = clf.feature_importances_
print("\n  Feature shift importance (top 8):")
imp_sorted = sorted(zip(FEATURE_COLS, importances), key=lambda r: -r[1])
for col, imp_v in imp_sorted[:8]:
    flag = "⚠️" if imp_v > 0.1 else "—"
    print(f"  {flag} {col}: {imp_v:.4f}")


# ============================================================
# T2: QUANTILE TRANSFORM + RIDGE
# ============================================================
section("T2 — QuantileTransformer + Ridge (outlier-robust)")


def cv_oof_test_with_transform(transformer_factory, model_factory, X, y, X_test, cv, winsor_fn=winsor):
    oof = np.zeros_like(y, dtype=float)
    test_fold_preds = []
    for tr, va in cv.split(X):
        y_tr = winsor_fn(y[tr]) if winsor_fn else y[tr]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("trans", transformer_factory()),
            ("m", model_factory()),
        ])
        pipe.fit(X[tr], y_tr)
        oof[va] = pipe.predict(X[va])
        test_fold_preds.append(pipe.predict(X_test))
    y_full = winsor_fn(y) if winsor_fn else y
    full_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("trans", transformer_factory()),
        ("m", model_factory()),
    ]).fit(X, y_full)
    return oof, np.mean(test_fold_preds, axis=0), full_pipe.predict(X_test)


results = []
print("\n[T2a] Quantile uniform → Ridge")
for alpha in [0.3, 1.0, 3.0, 10.0, 30.0]:
    oof, _, t_full = cv_oof_test_with_transform(
        lambda: QuantileTransformer(n_quantiles=500, output_distribution="uniform", random_state=SEED),
        lambda a=alpha: Ridge(alpha=a, random_state=SEED),
        X, y, X_test, cv)
    r2 = r2_score(y, oof)
    print(f"  Q-uniform + Ridge α={alpha}: OOF={r2:+.5f}")
    results.append((f"v13_quni_a{alpha}", r2, t_full))

print("\n[T2b] Quantile normal → Ridge")
for alpha in [0.3, 1.0, 3.0, 10.0, 30.0]:
    oof, _, t_full = cv_oof_test_with_transform(
        lambda: QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=SEED),
        lambda a=alpha: Ridge(alpha=a, random_state=SEED),
        X, y, X_test, cv)
    r2 = r2_score(y, oof)
    print(f"  Q-normal + Ridge α={alpha}: OOF={r2:+.5f}")
    results.append((f"v13_qnorm_a{alpha}", r2, t_full))


# ============================================================
# T3: BOOTSTRAP RIDGE ENSEMBLE (median)
# ============================================================
section("T3 — Bootstrap Ridge ensemble (median aggregation)")


def bootstrap_ridge(X, y, X_test, n_boot=200, alpha=1.0, seed=SEED, winsor_fn=winsor):
    """Train n_boot Ridge on bootstrap samples, return median test prediction + OOF-style estimate."""
    rng = np.random.default_rng(seed)
    n = len(X)
    test_preds = np.zeros((n_boot, len(X_test)))
    # OOB predictions for each train sample
    oob_preds = [[] for _ in range(n)]
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        oob_mask = ~np.isin(np.arange(n), idx)
        y_b = winsor_fn(y[idx]) if winsor_fn else y[idx]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha, random_state=seed + b)),
        ])
        pipe.fit(X[idx], y_b)
        test_preds[b] = pipe.predict(X_test)
        if oob_mask.sum() > 0:
            oob_p = pipe.predict(X[oob_mask])
            for j, oi in enumerate(np.where(oob_mask)[0]):
                oob_preds[oi].append(oob_p[j])
    test_med = np.median(test_preds, axis=0)
    test_mean = np.mean(test_preds, axis=0)
    # OOB pseudo-OOF
    oob_pred = np.array([np.median(p) if p else np.nan for p in oob_preds])
    valid = ~np.isnan(oob_pred)
    oob_r2 = r2_score(y[valid], oob_pred[valid])
    return oob_r2, test_med, test_mean


print("\n[T3] Bootstrap Ridge (n=200) for various α:")
for alpha in [0.3, 1.0, 3.0]:
    oob_r2, t_med, t_mean = bootstrap_ridge(X, y, X_test, n_boot=200, alpha=alpha)
    print(f"  α={alpha}: OOB R²={oob_r2:+.5f}  (using median aggregation)")
    results.append((f"v13_boot_median_a{alpha}", oob_r2, t_med))
    results.append((f"v13_boot_mean_a{alpha}", oob_r2, t_mean))


# ============================================================
# T4 (BONUS): Combination — Quantile + winsor target + Ridge
# ============================================================
section("T4 — Best Quantile setting + heavier regularization sweep")
print("\n[T4] Q-normal + finer α grid:")
for alpha in [0.5, 1.0, 2.0, 5.0]:
    for n_q in [100, 250, 500, 1000]:
        oof, _, t_full = cv_oof_test_with_transform(
            lambda nq=n_q: QuantileTransformer(n_quantiles=nq, output_distribution="normal", random_state=SEED),
            lambda a=alpha: Ridge(alpha=a, random_state=SEED),
            X, y, X_test, cv)
        r2 = r2_score(y, oof)
        print(f"  Q-norm(n={n_q}) + Ridge α={alpha}: OOF={r2:+.5f}")
        results.append((f"v13_qnorm{n_q}_a{alpha}", r2, t_full))


# ============================================================
section("🏆 RANKING — by OOF (top 10)")
results.sort(key=lambda r: -r[1])
for tag, r2, _ in results[:10]:
    marker = "🎯" if r2 > 0.033 else "—"
    print(f"  {marker} {tag:<40s} OOF={r2:+.5f}")


section("📤 SAVE TOP 3 CANDIDATES")
for tag, r2, test_pred in results[:3]:
    safe_tag = tag.replace(".", "p")
    fname = f"Baek_Seunghan_{safe_tag}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    out = SUB_DIR / fname
    sub.to_csv(out, index=False)
    print(f"  ✓ {fname:<55s} OOF={r2:+.5f}")

print("\n✅ Phase 13 done.")
