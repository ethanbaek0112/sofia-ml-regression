"""
Phase 19 — Final Genuinely Different Framings
==============================================
Phase 18 결과: Huber/X-winsor/stack/TheilSen/rank 모두 v7b 못 이김.
Residual std=1945 = pure noise → linear signal 다 뽑힘 입증.

마지막 시도 4가지 (sklearn 강의 범위 내):
F. PowerTransformer (YeoJohnson) on target — heavy-tail 대응
G. PLS Regression                          — latent space (Ridge와 다름)
H. Per-quantile models                     — 3개 모델 (top/mid/bot)
I. Magnitude-aware blend                   — v7b ± magnitude correction

통과 기준: OOF >= 0.033 AND corr(v7b) < 0.97
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
N_FOLDS = 5
FEATURE_COLS = [f"x{i}" for i in range(15)]


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def winsor_y(y, lo=0.5, hi=99.5):
    a, b = np.percentile(y, [lo, hi])
    return np.clip(y, a, b)


# ───────────────────────────────────────────────────────────────────
section("LOAD")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
X = train[FEATURE_COLS].values
y = train["target"].values
X_test = test[FEATURE_COLS].values
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
print(f"train={X.shape}, test={X_test.shape}")


# v7b OOF baseline
section("Baseline: v7b OOF")
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
oof_r2_v7b = r2_score(y, oof_v7b)
print(f"  v7b OOF R² = {oof_r2_v7b:+.5f}")


results = []  # (tag, oof_r2, corr_v7b, test_pred)


# ───────────────────────────────────────────────────────────────────
# F. PowerTransformer (YeoJohnson) on target
section("F. YeoJohnson target transform + Ridge")
for alpha in [0.3, 1.0, 3.0]:
    oof = np.zeros_like(y, dtype=float)
    test_preds = []
    for tr, va in cv.split(X):
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        y_tr_w = winsor_y(y[tr], 0.5, 99.5)
        y_tr_t = pt.fit_transform(y_tr_w.reshape(-1, 1)).ravel()
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha, random_state=42)),
        ])
        pipe.fit(X[tr], y_tr_t)
        pred_va = pipe.predict(X[va])
        pred_te = pipe.predict(X_test)
        # inverse transform
        oof[va] = pt.inverse_transform(pred_va.reshape(-1, 1)).ravel()
        test_preds.append(pt.inverse_transform(pred_te.reshape(-1, 1)).ravel())
    test_pred = np.mean(test_preds, axis=0)
    r2 = r2_score(y, oof)
    corr = np.corrcoef(oof, oof_v7b)[0, 1]
    print(f"  α={alpha}: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
    results.append((f"F_yj_a{alpha}", r2, corr, test_pred))


# ───────────────────────────────────────────────────────────────────
# G. PLS Regression
section("G. PLS Regression (latent components)")
for n_comp in [2, 4, 6, 8, 10, 13]:
    oof = np.zeros_like(y, dtype=float)
    test_preds = []
    for tr, va in cv.split(X):
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_tr = sc.fit_transform(imp.fit_transform(X[tr]))
        X_va = sc.transform(imp.transform(X[va]))
        X_te = sc.transform(imp.transform(X_test))
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_tr, winsor_y(y[tr], 0.5, 99.5))
        oof[va] = pls.predict(X_va).ravel()
        test_preds.append(pls.predict(X_te).ravel())
    test_pred = np.mean(test_preds, axis=0)
    r2 = r2_score(y, oof)
    corr = np.corrcoef(oof, oof_v7b)[0, 1]
    print(f"  n_comp={n_comp:>2}: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
    results.append((f"G_pls_n{n_comp}", r2, corr, test_pred))


# ───────────────────────────────────────────────────────────────────
# H. Per-quantile models (3 separate Ridges)
section("H. Per-quantile target stratification")
# Split y into 3 quantiles, train separate Ridge each, gate by predicted v7b magnitude
y_q33, y_q67 = np.percentile(y, [33, 67])
print(f"  q33={y_q33:.1f}, q67={y_q67:.1f}")
oof = np.zeros_like(y, dtype=float)
test_preds_list = []
for tr, va in cv.split(X):
    # Train 3 ridges
    masks_tr = [
        y[tr] <= y_q33,
        (y[tr] > y_q33) & (y[tr] <= y_q67),
        y[tr] > y_q67,
    ]
    pipes = []
    for m in masks_tr:
        if m.sum() < 50:
            pipes.append(None)
            continue
        p = v7b_pipe()
        p.fit(X[tr][m], winsor_y(y[tr][m], 0.5, 99.5))
        pipes.append(p)
    # Get v7b prediction to gate (use this fold's v7b only — leak-free)
    v7b_fold = v7b_pipe()
    v7b_fold.fit(X[tr], winsor_y(y[tr], 0.5, 99.5))
    gate_va = v7b_fold.predict(X[va])
    gate_te = v7b_fold.predict(X_test)
    # Predict each row from its gated model
    for i, (g, val_idx) in enumerate(zip(gate_va, va)):
        if g <= y_q33:
            k = 0
        elif g <= y_q67:
            k = 1
        else:
            k = 2
        oof[val_idx] = pipes[k].predict(X[val_idx:val_idx + 1])[0] if pipes[k] else gate_va[i]
    test_p = np.zeros(len(X_test))
    for i, g in enumerate(gate_te):
        if g <= y_q33:
            k = 0
        elif g <= y_q67:
            k = 1
        else:
            k = 2
        test_p[i] = pipes[k].predict(X_test[i:i + 1])[0] if pipes[k] else gate_te[i]
    test_preds_list.append(test_p)
test_pred = np.mean(test_preds_list, axis=0)
r2 = r2_score(y, oof)
corr = np.corrcoef(oof, oof_v7b)[0, 1]
print(f"  Per-quantile: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
results.append(("H_perquantile", r2, corr, test_pred))


# ───────────────────────────────────────────────────────────────────
# I. Magnitude-aware blend (different gain for high/low predictions)
section("I. Magnitude-aware piecewise scaling on v7b")
# Hypothesis: maybe v7b under-predicts in tails but overpredicts in middle
# Apply piecewise scaling: small predictions kept, large predictions scaled
for scale_high in [1.05, 1.10, 1.20, 1.50]:
    # Apply only to predictions in top/bottom 20% magnitude
    threshold = np.percentile(np.abs(oof_v7b), 80)
    mask_tail = np.abs(oof_v7b) > threshold
    oof_adj = oof_v7b.copy()
    oof_adj[mask_tail] *= scale_high
    test_threshold = np.percentile(np.abs(test_v7b), 80)
    test_mask = np.abs(test_v7b) > test_threshold
    test_adj = test_v7b.copy()
    test_adj[test_mask] *= scale_high
    r2 = r2_score(y, oof_adj)
    corr = np.corrcoef(oof_adj, oof_v7b)[0, 1]
    print(f"  tail×{scale_high}: OOF R² = {r2:+.5f}, corr(v7b) = {corr:.4f}")
    results.append((f"I_magtail_{scale_high}", r2, corr, test_adj))


# ───────────────────────────────────────────────────────────────────
# Summary
section("SUMMARY — sorted by OOF R²")
results_sorted = sorted(results, key=lambda r: -r[1])
print(f"{'Tag':<28} {'OOF R²':>10} {'corr(v7b)':>11} {'Verdict':>22}")
print("─" * 75)
for tag, r2, corr, _ in results_sorted:
    if r2 < 0.025:
        verdict = "❌ weak OOF"
    elif corr > 0.99:
        verdict = "🟡 too similar to v7b"
    elif corr < 0.95 and r2 > 0.033:
        verdict = "✅✅ NEW SIGNAL!"
    elif r2 > 0.034:
        verdict = "🟢 strong OOF"
    elif corr < 0.97:
        verdict = "🟡 some diversity"
    else:
        verdict = "🟡 modest"
    print(f"{tag:<28} {r2:>+10.5f} {corr:>11.4f}  {verdict:>22}")


# Save promising
section("BUILD CANDIDATE SUBMISSIONS (OOF >= 0.033 AND corr < 0.97)")
promising = [(tag, r2, corr, tp) for tag, r2, corr, tp in results
             if r2 >= 0.033 and corr < 0.97]
if not promising:
    print("  ⚠️ No new signal found. v7b remains the champion.")
    # Also save the highest-OOF candidates (regardless of corr) as backup
    backup = sorted(results, key=lambda r: -r[1])[:2]
    print(f"\n  📦 Saving top-2 OOF candidates as backup (may still be valuable):")
    for tag, r2, corr, tp in backup:
        sub = pd.DataFrame({"Id": test["Id"].values, "target": tp})
        sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
        out = SUBMISSIONS_DIR / f"Baek_Seunghan_v19_{tag}.csv"
        sub.to_csv(out, index=False)
        print(f"     {out.name}  (OOF={r2:+.5f}, corr={corr:.4f})")
else:
    for tag, r2, corr, tp in promising:
        sub = pd.DataFrame({"Id": test["Id"].values, "target": tp})
        sub = sub.set_index("Id").loc[sample["Id"]].reset_index()
        out = SUBMISSIONS_DIR / f"Baek_Seunghan_v19_{tag}.csv"
        sub.to_csv(out, index=False)
        print(f"  ✅ {out.name}")
        print(f"     OOF={r2:+.5f}, corr(v7b)={corr:.4f}")

print("\n✅ Phase 19 done.")
