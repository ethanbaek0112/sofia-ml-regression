"""
Build final two Phase-8 submissions:
  v08a (SAFE):   Winsor[0.3-99.7] + Ridge α=0.3  — OOF +0.0357
  v08b (RISKY):  Winsor[0.2-99.8] + Ridge α=0.3  — OOF +0.0381

도박 (v08b)이 v7d 데자뷰 위험. 안전 (v08a)이 sweet spot 살짝 바깥.
둘 다 던져서 calibration gap이 [0.5-99.5] 너머에서 어떻게 변하는지 데이터 수집.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
FEATURE_COLS = [f"x{i}" for i in range(15)]
SEED = 42


def build(lo, hi, alpha, tag, label):
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    X_tr = train[FEATURE_COLS].values
    y_tr = train["target"].values
    X_te = test[FEATURE_COLS].values

    a, b = np.percentile(y_tr, [lo, hi])
    y_clipped = np.clip(y_tr, a, b)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=alpha, random_state=SEED)),
    ]).fit(X_tr, y_clipped)

    pred = pipe.predict(X_te)
    print(f"\n{label}  ({tag})")
    print(f"  Winsor=[{lo}, {hi}]  →  clip range [{a:+.1f}, {b:+.1f}]")
    print(f"  Ridge α={alpha}")
    print(f"  Test pred: mean={pred.mean():+.2f}  std={pred.std():.2f}  "
          f"min={pred.min():+.2f}  max={pred.max():+.2f}")

    sub = pd.DataFrame({"Id": test["Id"].values, "target": pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    out = SUBMISSIONS_DIR / f"Baek_Seunghan_{tag}.csv"
    sub.to_csv(out, index=False)
    print(f"  → {out.name}")
    return out


print("=" * 70)
print("PHASE 8 FINAL SUBMISSIONS — 둘 다 던지기")
print("=" * 70)

a_path = build(0.3, 99.7, 0.3,
               tag="v08a_safe_winsor03to997_ridge_a03",
               label="🛡️  v08a SAFE — sweet spot 살짝 바깥")

b_path = build(0.2, 99.8, 0.3,
               tag="v08b_risky_winsor02to998_ridge_a03",
               label="🎲 v08b RISKY — v7d 데자뷰 zone (variance high)")

# v7b 와 비교
v7b_path = SUBMISSIONS_DIR / "Baek_Seunghan_v07b_winsor05to995+ridge_a1.csv"

print("\n" + "=" * 70)
print("PREDICTION 비교 (v7b 기준)")
print("=" * 70)
df_v7b = pd.read_csv(v7b_path).set_index("Id")["target"]
df_a = pd.read_csv(a_path).set_index("Id")["target"]
df_b = pd.read_csv(b_path).set_index("Id")["target"]

for label, df in [("v08a vs v7b", df_a - df_v7b), ("v08b vs v7b", df_b - df_v7b),
                  ("v08b vs v08a", df_b - df_a)]:
    print(f"  {label}: mean diff={df.mean():+.3f}  std={df.std():.3f}  "
          f"max|diff|={df.abs().max():.2f}")

print("\n✅ Both submissions ready. Go submit on Kaggle (5/day limit).")
