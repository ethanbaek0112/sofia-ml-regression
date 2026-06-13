"""
Phase 8: Sweet spot 정밀 탐색 — Pawel 추월 작전
======================================================
v7b (Winsor[0.5-99.5] + Ridge α=1) → LB 0.04843, OOF +0.0327
Pawel은 0.04877 (우리보다 +0.00034). Public LB 51%만 사용이라 노이즈 수준.

전략:
1. Symmetric fine sweep — sweet spot 진짜 어디인지
2. Asymmetric winsor — target이 left-skewed (p1=-2655 vs p99=+1591)
   → negative tail을 더 공격적으로 clip하는 게 합리적일 수 있음
3. Ridge α 더 미세하게 (0.3 ~ 5)

각 후보 OOF로 평가 → Top 2-3개만 실제 제출.
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

SEED = 42
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]
SUBMISSION_NAME = "Baek_Seunghan"


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def winsor_filter(lo: float, hi: float):
    def f(X, y):
        a, b = np.percentile(y, [lo, hi])
        return X, np.clip(y, a, b)
    return f


def ridge_pipeline_factory(alpha: float):
    """Return a *factory* — fresh pipeline per fold (no leakage)."""
    def fac():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha, random_state=SEED)),
        ])
    return fac


def cv_oof_filtered(fac, X, y, cv, y_filter):
    fold_scores, oof = [], np.zeros_like(y, dtype=float)
    for tr, va in cv.split(X):
        X_tr, y_tr = y_filter(X[tr], y[tr])
        pipe = fac().fit(X_tr, y_tr)
        pred = pipe.predict(X[va])
        oof[va] = pred
        fold_scores.append(r2_score(y[va], pred))
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), float(r2_score(y, oof))


def make_short_name(lo: float, hi: float, alpha: float) -> str:
    return (f"winsor{lo}to{hi}_ridge_a{alpha}"
            .replace(".", "")
            .replace("-", "neg"))


# ───────────────────────────── LOAD ─────────────────────────────
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X_train = train[FEATURE_COLS].values
y_train = train["target"].values
X_test = test[FEATURE_COLS].values
test_ids = test["Id"].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Quick skewness check (printed for the record)
from scipy.stats import skew, kurtosis  # type: ignore
print(f"Target: n={len(y_train)}, skew={skew(y_train):+.3f}, kurt={kurtosis(y_train):+.3f}")
print(f"  p0.5={np.percentile(y_train, 0.5):+.1f}, "
      f"p99.5={np.percentile(y_train, 99.5):+.1f}  "
      f"→ negative tail {abs(np.percentile(y_train, 0.5))/np.percentile(y_train, 99.5):.2f}× heavier")


results = []


def run(label: str, lo: float, hi: float, alpha: float):
    fac = ridge_pipeline_factory(alpha)
    cv_m, cv_s, oof = cv_oof_filtered(fac, X_train, y_train, cv, winsor_filter(lo, hi))
    print(f"  {label}: CV={cv_m:+.5f}±{cv_s:.5f}  OOF={oof:+.5f}")
    results.append({
        "name": label, "lo": lo, "hi": hi, "alpha": alpha,
        "cv_mean": cv_m, "cv_std": cv_s, "oof_r2": oof,
        "factory": fac, "y_filter": winsor_filter(lo, hi),
    })


# ──────────────── BLOCK 1: SYMMETRIC FINE SWEEP ────────────────
section("BLOCK 1 — Symmetric fine sweep")
SYM_BANDS = [(0.2, 99.8), (0.3, 99.7), (0.4, 99.6),
             (0.5, 99.5), (0.6, 99.4), (0.7, 99.3), (0.8, 99.2)]
ALPHAS = [0.3, 0.5, 1.0, 2.0, 5.0]

for lo, hi in SYM_BANDS:
    for a in ALPHAS:
        run(f"sym[{lo:>4}-{hi}]+Ridge α={a:<4}", lo, hi, a)


# ──────────────── BLOCK 2: ASYMMETRIC — CLIP NEG MORE ────────────────
# Target left-skewed → negative tail heavier. Clip more aggressively on lo side.
section("BLOCK 2 — Asymmetric: clip NEGATIVE side more aggressively")
for lo, hi in [(0.7, 99.5), (1.0, 99.5), (1.5, 99.5), (2.0, 99.5),
               (1.0, 99.7), (1.5, 99.7)]:
    run(f"asym[{lo:>4}-{hi}]+Ridge α=1", lo, hi, 1.0)


# ──────────────── BLOCK 3: ASYMMETRIC — CLIP POS MORE ────────────────
# Counter-hypothesis: maybe positive side prediction is harder, clip pos more.
section("BLOCK 3 — Asymmetric: clip POSITIVE side more aggressively")
for lo, hi in [(0.5, 99.3), (0.5, 99.0), (0.5, 98.5),
               (0.3, 99.0), (0.3, 98.5)]:
    run(f"asym[{lo:>4}-{hi}]+Ridge α=1", lo, hi, 1.0)


# ──────────────── BLOCK 4: TIGHTEN α AROUND WINNER ────────────────
section("BLOCK 4 — Fine α sweep at 0.5-99.5 (current winner)")
for a in [0.1, 0.2, 0.7, 1.5, 3.0]:
    run(f"sweet[0.5-99.5]+Ridge α={a:<4}", 0.5, 99.5, a)


# ──────────────── REPORT ────────────────
section("TOP 20 by OOF R²")
df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("factory", "y_filter")}
    for r in results
]).sort_values("oof_r2", ascending=False).reset_index(drop=True)
print(df.head(20).to_string(index=False))

df.to_csv(ROOT / "docs" / "phase8_results.csv", index=False)
print(f"\nSaved: docs/phase8_results.csv  ({len(df)} total experiments)")


# ──────────────── BUILD TOP-3 SUBMISSIONS ────────────────
section("BUILD TOP-3 CANDIDATE SUBMISSIONS")
for rank in range(min(3, len(df))):
    name = df.iloc[rank]["name"]
    rec = next(r for r in results if r["name"] == name)
    pipe = rec["factory"]()
    X_fit, y_fit = rec["y_filter"](X_train, y_train)
    pipe.fit(X_fit, y_fit)
    pred = pipe.predict(X_test)

    short = make_short_name(rec["lo"], rec["hi"], rec["alpha"])
    out = SUBMISSIONS_DIR / f"{SUBMISSION_NAME}_v08_{rank+1:02d}_{short}.csv"
    sub = pd.DataFrame({"Id": test_ids, "target": pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    sub.to_csv(out, index=False)

    expected_lb = rec["oof_r2"] + 0.0157  # calibration gap observed on v7b
    pawel = 0.04877
    margin = expected_lb - pawel
    print(f"  #{rank+1}  {name}")
    print(f"        OOF={rec['oof_r2']:+.5f}  →  Expected LB ≈ {expected_lb:+.5f}  "
          f"(vs Pawel {pawel:+.5f}: {'+' if margin >= 0 else ''}{margin:+.5f})")
    print(f"        →  {out.name}")

print("\n✅ Phase 8 done. Pick top 1-2 to actually submit (5/day limit).")
