"""
Phase 14: Diagnosis — Why are we stuck at LB ≈ 0.048?
=============================================================================
관찰: Ridge/Winsor/Quantile family 모두 LB ≈ 0.0484로 수렴.
     상위 4명은 0.050+ 달성. 우리가 뭘 놓치고 있나?

진단 항목:
  1. Target 분포 (skew, multimodality, outlier 패턴)
  2. Feature-target 관계 (선형 vs 비선형 sniffing)
  3. Ridge prediction의 잔차 패턴 (어떤 sample에서 틀리나)
  4. 우리 best predictions(v7b) vs v11a(다른 영역에서 강한가?)
  5. Target과 prediction의 분포 비교 (calibration 필요한가?)
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


# LOAD
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# ============================================================
section("D1 — Target distribution")
print(f"  N = {len(y)}")
print(f"  mean = {y.mean():+.4f}")
print(f"  std  = {y.std():+.4f}")
print(f"  min  = {y.min():+.4f}")
print(f"  max  = {y.max():+.4f}")
print(f"  median = {np.median(y):+.4f}")
print(f"  skew (rough) = {((y - y.mean()) ** 3).mean() / y.std() ** 3:+.4f}")
print(f"  kurtosis (rough) = {((y - y.mean()) ** 4).mean() / y.std() ** 4 - 3:+.4f}")
print(f"\n  Percentiles:")
for p in [0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]:
    print(f"    p{p:>5}: {np.percentile(y, p):+.4f}")

# Check for special values (zeros, negatives, etc.)
print(f"\n  # zeros: {(y == 0).sum()}")
print(f"  # negatives: {(y < 0).sum()}")
print(f"  # positive: {(y > 0).sum()}")


# ============================================================
section("D2 — Feature value ranges (any weird ones?)")
print(f"  {'feat':<5s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s} {'NaN%':>6s} {'|corr|':>8s}")
for i, col in enumerate(FEATURE_COLS):
    xi = X[:, i]
    nan_pct = np.isnan(xi).mean() * 100
    xi_clean = xi[~np.isnan(xi)]
    y_clean = y[~np.isnan(xi)]
    corr = abs(np.corrcoef(xi_clean, y_clean)[0, 1])
    print(f"  {col:<5s} {xi_clean.min():>+10.3f} {xi_clean.max():>+10.3f} {xi_clean.mean():>+10.3f} {xi_clean.std():>+10.3f} {nan_pct:>5.1f}% {corr:>8.4f}")


# ============================================================
section("D3 — Ridge v7b: where does it fail?")
# Get OOF predictions
oof = np.zeros_like(y)
for tr, va in cv.split(X):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=1.0, random_state=SEED)),
    ])
    pipe.fit(X[tr], winsor(y[tr]))
    oof[va] = pipe.predict(X[va])

residual = y - oof
print(f"  OOF R² = {r2_score(y, oof):+.5f}")
print(f"  residual stats: mean={residual.mean():+.4f}, std={residual.std():+.4f}")
print(f"  residual percentiles:")
for p in [0.5, 5, 25, 50, 75, 95, 99.5]:
    print(f"    p{p:>5}: {np.percentile(residual, p):+.4f}")

# By target magnitude
print(f"\n  R² by target quintile (where do we fail?):")
quintile_edges = np.percentile(y, [0, 20, 40, 60, 80, 100])
for i in range(5):
    lo, hi = quintile_edges[i], quintile_edges[i + 1]
    mask = (y >= lo) & (y < hi) if i < 4 else (y >= lo) & (y <= hi)
    if mask.sum() > 0:
        r2 = r2_score(y[mask], oof[mask]) if mask.sum() > 1 else float('nan')
        bias = (oof[mask] - y[mask]).mean()
        print(f"    Q{i+1} y∈[{lo:+.3f},{hi:+.3f}] n={mask.sum()} R²={r2:+.5f} bias={bias:+.4f}")


# ============================================================
section("D4 — Train target vs Test prediction distribution (calibration check)")
# v7b full pipeline
full_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("m", Ridge(alpha=1.0, random_state=SEED)),
]).fit(X, winsor(y))
test_pred_v7b = full_pipe.predict(X_test)

print(f"  Train target — mean={y.mean():+.4f} std={y.std():+.4f}")
print(f"  v7b OOF — mean={oof.mean():+.4f} std={oof.std():+.4f}  (compressed std = Ridge shrinkage)")
print(f"  v7b test pred — mean={test_pred_v7b.mean():+.4f} std={test_pred_v7b.std():+.4f}")
print(f"\n  Std ratio (target/pred): {y.std() / test_pred_v7b.std():.3f}x")
print("  → If >> 1, predictions are too compressed. Rescaling MAY help LB.")


# ============================================================
section("D5 — Test prediction percentiles (do they look like train target?)")
print(f"  {'p':<6s} {'train_y':>10s} {'test_pred':>10s} {'ratio':>8s}")
for p in [1, 5, 25, 50, 75, 95, 99]:
    yp = np.percentile(y, p)
    tp = np.percentile(test_pred_v7b, p)
    ratio = yp / tp if tp != 0 else float('nan')
    print(f"  {p:>5}%  {yp:>+10.4f} {tp:>+10.4f} {ratio:>8.3f}")


# ============================================================
section("D6 — Quick test: rescale predictions to match train target std")
# Idea: if Ridge compresses predictions, multiply by ratio to match target std
ratio = y.std() / test_pred_v7b.std()
test_pred_rescaled = test_pred_v7b * ratio
print(f"  Original test_pred std: {test_pred_v7b.std():.4f}")
print(f"  Rescaled (×{ratio:.3f}) std: {test_pred_rescaled.std():.4f}")
print(f"  Target std: {y.std():.4f}")

# How would rescaling affect OOF?
oof_rescaled_mean = oof * ratio
print(f"\n  OOF R² original   = {r2_score(y, oof):+.5f}")
print(f"  OOF R² rescaled   = {r2_score(y, oof_rescaled_mean):+.5f}  (sanity check)")
print("  → On OOF rescaling hurts because we're predicting the SAME data.")
print("  → On test/LB it MIGHT help if test distribution differs from compressed pred.")
print("  (실험 가치: 던질 만한 후보)")

# Save rescaled candidate
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred_rescaled})
sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
out = SUB_DIR / "Baek_Seunghan_v14_rescale_v7b.csv"
sub.to_csv(out, index=False)
print(f"\n  Saved: {out.name}  (v7b × {ratio:.3f})")


# ============================================================
section("✅ Diagnosis done — see analysis above")
print("""
KEY QUESTIONS TO ANSWER FROM OUTPUT:
  - D1: Is target heavily skewed / bimodal? → log/Box-Cox candidate
  - D2: Any feature with extreme range or high NaN%? → custom handling
  - D3: Where does Ridge fail (head, tail, middle of target)? → asymmetric model
  - D4: Are predictions compressed (low std)? → rescaling helps?
  - D5: Test pred distribution match? → calibration?
""")
