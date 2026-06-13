"""
Phase 20 — Magtrim Symmetry Test
==================================
Hypothesis: v7b's tails may be over-extended.
- magtail ×1.2 (expand) → LB -0.001
- magtrim ×0.9 (shrink) → LB +0.001? (symmetric counterpart)

Pure symmetry test, NOT OOF optimization.
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
SUBMISSIONS_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]


def section(t):
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")


def winsor_y(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


# Load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


# v7b OOF + test
section("v7b baseline")
v7b_pipe = lambda: Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("m", Ridge(alpha=1.0, random_state=42)),
])
oof_v7b = np.zeros_like(y, dtype=float)
test_v7b_list = []
for tr, va in cv.split(X):
    p = v7b_pipe()
    p.fit(X[tr], winsor_y(y[tr], 0.5, 99.5))
    oof_v7b[va] = p.predict(X[va])
    test_v7b_list.append(p.predict(X_test))
test_v7b = np.mean(test_v7b_list, axis=0)
print(f"  v7b OOF R² = {r2_score(y, oof_v7b):+.5f}")


# Magtrim sweep — symmetric counterparts to magtail
section("Magtrim sweep (shrink top 20% magnitude)")
threshold_oof = np.percentile(np.abs(oof_v7b), 80)
threshold_test = np.percentile(np.abs(test_v7b), 80)

candidates = {}
for trim_factor in [0.95, 0.90, 0.85, 0.80, 0.70]:
    oof_adj = oof_v7b.copy()
    mask_oof = np.abs(oof_v7b) > threshold_oof
    oof_adj[mask_oof] *= trim_factor
    test_adj = test_v7b.copy()
    mask_test = np.abs(test_v7b) > threshold_test
    test_adj[mask_test] *= trim_factor
    r2 = r2_score(y, oof_adj)
    corr = np.corrcoef(oof_adj, oof_v7b)[0, 1]
    print(f"  ×{trim_factor:.2f}: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
    candidates[trim_factor] = test_adj


# Save the primary candidate (×0.9 symmetric to magtail ×1.2)
section("BUILD")
for trim_factor in [0.90, 0.80]:
    test_pred = candidates[trim_factor]
    sub = pd.DataFrame({"Id": test["Id"].values, "target": test_pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    tag = f"v20_magtrim_{str(trim_factor).replace('.', 'p')}"
    out = SUBMISSIONS_DIR / f"Baek_Seunghan_{tag}.csv"
    sub.to_csv(out, index=False)
    print(f"  ✅ {out.name}")

print("\n✅ Phase 20 done.")
