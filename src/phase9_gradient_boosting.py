"""
Phase 9: Gradient Boosting 투입 — kelly fu/Pawel 추격
=========================================================
과제 문서가 명시적으로 LightGBM/XGBoost를 "표준 워크플로"로 가정.
강의 범위 self-imposed 제약 해제. 단, winsorize 인사이트는 활용 가능.

이번 phase는 sklearn HGBR로 baseline (LightGBM/XGBoost 설치 동안).
HGBR는 LightGBM과 사실상 동일 알고리즘.

실험 블록:
  1. HGBR default + 단순 하이퍼 sweep (NO winsor) — gradient boosting raw 성능
  2. HGBR + winsor[0.5-99.5] (강의 인사이트 적용)
  3. HGBR + winsor band sweep (tree는 winsor 효과 다를 수 있음)
  4. HGBR loss='absolute_error' (built-in outlier robustness)

발표 스토리:
  Part 1 — 강의 범위 + winsor sweet spot U-curve 발견
  Part 2 — 실전 도구 (HGBR/LightGBM) + 같은 인사이트 재적용
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
SEED = 42
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]

# Reference points
KELLY_FU = 0.04972
PAWEL = 0.04877
V7B_LB = 0.04843


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def winsor_filter(lo, hi):
    def f(X, y):
        a, b = np.percentile(y, [lo, hi])
        return X, np.clip(y, a, b)
    return f


def identity_filter(X, y):
    return X, y


def hgbr_pipeline(**kw):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingRegressor(random_state=SEED, **kw)),
    ])


def cv_oof_filtered(pipe_fac, X, y, cv, y_filter):
    fold_scores, oof = [], np.zeros_like(y, dtype=float)
    for tr, va in cv.split(X):
        X_tr, y_tr = y_filter(X[tr], y[tr])
        pipe = pipe_fac().fit(X_tr, y_tr)
        pred = pipe.predict(X[va])
        oof[va] = pred
        fold_scores.append(r2_score(y[va], pred))
    return (
        float(np.mean(fold_scores)),
        float(np.std(fold_scores)),
        float(r2_score(y, oof)),
    )


# LOAD
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X_train = train[FEATURE_COLS].values
y_train = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

results = []


def run(label, hp, y_filter=identity_filter):
    fac = lambda: hgbr_pipeline(**hp)
    cv_m, cv_s, oof = cv_oof_filtered(fac, X_train, y_train, cv, y_filter)
    print(f"  {label:55s}  CV={cv_m:+.5f}  OOF={oof:+.5f}")
    results.append({
        "name": label, "hp": hp,
        "cv_mean": cv_m, "cv_std": cv_s, "oof_r2": oof,
        "y_filter": y_filter,
    })


# BLOCK 1: HGBR raw (NO winsor)
section("BLOCK 1 — HGBR baseline (강의 외 모델이 raw로 어디까지?)")
run("HGBR default", {})
run("HGBR lr=0.05 iter=300", dict(learning_rate=0.05, max_iter=300))
run("HGBR lr=0.1 iter=200", dict(learning_rate=0.1, max_iter=200))
run("HGBR lr=0.03 iter=500", dict(learning_rate=0.03, max_iter=500))
run("HGBR deep leaf=63", dict(max_leaf_nodes=63))
run("HGBR shallow leaf=15", dict(max_leaf_nodes=15))
run("HGBR strong-reg min_samples=50", dict(min_samples_leaf=50))


# BLOCK 2: HGBR + winsor sweet spot
section("BLOCK 2 — HGBR + Winsor[0.5-99.5] (강의 인사이트 적용)")
sweet = winsor_filter(0.5, 99.5)
run("HGBR-sweet default", {}, sweet)
run("HGBR-sweet lr=0.05 iter=300", dict(learning_rate=0.05, max_iter=300), sweet)
run("HGBR-sweet lr=0.03 iter=500", dict(learning_rate=0.03, max_iter=500), sweet)
run("HGBR-sweet shallow leaf=15", dict(max_leaf_nodes=15), sweet)
run("HGBR-sweet leaf=15 lr=0.05", dict(max_leaf_nodes=15, learning_rate=0.05, max_iter=300), sweet)
run("HGBR-sweet strong-reg leaf=8", dict(max_leaf_nodes=8, min_samples_leaf=30), sweet)


# BLOCK 3: HGBR + winsor band sweep
section("BLOCK 3 — HGBR + Winsor band sweep")
for lo, hi in [(0.3, 99.7), (1.0, 99.0), (2.0, 98.0), (5.0, 95.0)]:
    run(f"HGBR + winsor[{lo}-{hi}]",
        dict(learning_rate=0.05, max_iter=300),
        winsor_filter(lo, hi))


# BLOCK 4: Huber-like loss for outlier robustness
section("BLOCK 4 — HGBR loss='absolute_error' (built-in robust)")
run("HGBR loss=absolute_error", dict(loss="absolute_error"))
run("HGBR loss=absolute_error sweet", dict(loss="absolute_error"), sweet)


# REPORT
section("TOP 20 by OOF R²")
df = pd.DataFrame([
    {"name": r["name"], "cv_mean": r["cv_mean"],
     "cv_std": r["cv_std"], "oof_r2": r["oof_r2"]}
    for r in results
]).sort_values("oof_r2", ascending=False).reset_index(drop=True)
print(df.head(20).to_string(index=False))
df.to_csv(ROOT / "docs" / "phase9_results.csv", index=False)
print(f"\nSaved: docs/phase9_results.csv  ({len(df)} total)")


# BUILD TOP-3 SUBMISSIONS
section("BUILD TOP-3 CANDIDATE SUBMISSIONS")
for rank in range(min(3, len(df))):
    name = df.iloc[rank]["name"]
    rec = next(r for r in results if r["name"] == name)

    pipe = hgbr_pipeline(**rec["hp"])
    X_fit, y_fit = rec["y_filter"](X_train, y_train)
    pipe.fit(X_fit, y_fit)
    pred = pipe.predict(X_test)

    # Trees often have small calibration gap (~0 to +0.005)
    expected_low = rec["oof_r2"] + 0.000
    expected_high = rec["oof_r2"] + 0.005

    print(f"\n  #{rank + 1}  {name}")
    print(f"        OOF={rec['oof_r2']:+.5f}  →  Expected LB ≈ "
          f"[{expected_low:+.4f}, {expected_high:+.4f}]")
    for tag, ref in [("kelly_fu", KELLY_FU), ("Pawel", PAWEL), ("v7b", V7B_LB)]:
        margin = expected_high - ref
        status = "✅ AHEAD" if margin > 0 else "❌ BEHIND"
        print(f"        vs {tag:9s} {ref}: {status} ({margin:+.4f})")

    short = (name.lower()
             .replace(" ", "_").replace("=", "").replace("[", "")
             .replace("]", "").replace(".", "").replace(",", "_")
             .replace("-", "to").replace("+", "_").replace("/", "_"))[:60]
    out = SUBMISSIONS_DIR / f"Baek_Seunghan_v09_{rank + 1:02d}_{short}.csv"
    sub = pd.DataFrame({"Id": test["Id"].values, "target": pred})
    sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
    sub.to_csv(out, index=False)
    print(f"        → {out.name}")

print("\n✅ Phase 9 done. Pick top 1-2 to submit (남은 daily 한도 안에서).")
