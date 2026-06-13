"""
Exploratory Data Analysis (EDA) — Sofia ML Regression
======================================================

Purpose:
- Understand the shape, types, missingness of train/test
- Inspect target distribution (for Lec 07 normality assumption check)
- Compute feature statistics + correlation with target
- Save key plots to docs/eda_plots/ for the presentation

All terminology aligned with Petiushko lectures.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "docs" / "eda_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [f"x{i}" for i in range(15)]
TARGET_COL = "target"
ID_COL = "Id"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
section("1. LOAD DATA")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"train shape: {train.shape}")
print(f"test  shape: {test.shape}")
print(f"sample_submission shape: {sample_sub.shape}")
print(f"\ntrain columns: {list(train.columns)}")
print(f"test  columns: {list(test.columns)}")


# ---------------------------------------------------------------------------
# 2. ID overlap check (paranoia)
# ----------------------------------------------------------------
section("2. ID OVERLAP CHECK")
train_ids = set(train[ID_COL])
test_ids = set(test[ID_COL])
overlap = train_ids & test_ids
print(f"train IDs: {len(train_ids)}, test IDs: {len(test_ids)}")
print(f"overlap : {len(overlap)} (expect 0)")
print(f"sample_submission first 5 IDs: {sample_sub[ID_COL].head().tolist()}")
print(f"test IDs match sample_submission IDs? "
      f"{set(test[ID_COL]) == set(sample_sub[ID_COL])}")


# ---------------------------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------------------------
section("3. MISSING VALUES")
missing_train = train[FEATURE_COLS].isnull().sum()
missing_test = test[FEATURE_COLS].isnull().sum()
missing_df = pd.DataFrame({"train_NaN": missing_train, "test_NaN": missing_test})
missing_df["train_pct"] = (missing_df["train_NaN"] / len(train) * 100).round(2)
missing_df["test_pct"] = (missing_df["test_NaN"] / len(test) * 100).round(2)
print(missing_df)
print(f"\nrows in train with at least one NaN: "
      f"{train[FEATURE_COLS].isnull().any(axis=1).sum()} / {len(train)}")
print(f"rows in test  with at least one NaN: "
      f"{test[FEATURE_COLS].isnull().any(axis=1).sum()} / {len(test)}")


# ---------------------------------------------------------------------------
# 4. Target distribution (Lec 07 slide 11: normality assumption)
# ---------------------------------------------------------------------------
section("4. TARGET DISTRIBUTION")
y = train[TARGET_COL]
print(f"target stats:")
print(f"  count : {len(y)}")
print(f"  mean  : {y.mean():.4f}")
print(f"  std   : {y.std():.4f}")
print(f"  min   : {y.min():.4f}")
print(f"  25%   : {y.quantile(0.25):.4f}")
print(f"  50%   : {y.median():.4f}")
print(f"  75%   : {y.quantile(0.75):.4f}")
print(f"  max   : {y.max():.4f}")
print(f"  skew  : {y.skew():.4f}  (|skew|<0.5 = roughly symmetric)")
print(f"  kurt  : {y.kurt():.4f}  (>3 = heavy tails)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=50, color="#0053e2", edgecolor="white")
axes[0].set_title("Target distribution (histogram)")
axes[0].set_xlabel("target")
axes[0].set_ylabel("frequency")
axes[1].boxplot(y, vert=False)
axes[1].set_title("Target distribution (boxplot)")
axes[1].set_xlabel("target")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_target_distribution.png", bbox_inches="tight")
plt.close()
print(f"\nSaved: {PLOTS_DIR / '01_target_distribution.png'}")


# ---------------------------------------------------------------------------
# 5. Feature distributions
# ---------------------------------------------------------------------------
section("5. FEATURE STATISTICS")
print(train[FEATURE_COLS].describe().T.round(4))

fig, axes = plt.subplots(3, 5, figsize=(18, 9))
for i, col in enumerate(FEATURE_COLS):
    ax = axes[i // 5][i % 5]
    ax.hist(train[col].dropna(), bins=40, color="#0053e2", alpha=0.7, label="train")
    ax.hist(test[col].dropna(), bins=40, color="#ffc220", alpha=0.5, label="test")
    ax.set_title(col)
    if i == 0:
        ax.legend(fontsize=8)
plt.suptitle("Feature distributions — train (blue) vs test (yellow)", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_feature_distributions.png", bbox_inches="tight")
plt.close()
print(f"\nSaved: {PLOTS_DIR / '02_feature_distributions.png'}")


# ---------------------------------------------------------------------------
# 6. Correlation with target
# ---------------------------------------------------------------------------
section("6. CORRELATION WITH TARGET")
# Drop rows with any NaN for clean correlation
train_clean = train.dropna(subset=FEATURE_COLS + [TARGET_COL])
corr_with_target = (
    train_clean[FEATURE_COLS + [TARGET_COL]]
    .corr()[TARGET_COL]
    .drop(TARGET_COL)
    .sort_values(key=abs, ascending=False)
)
print("Pearson correlation with target (sorted by |corr|):")
print(corr_with_target.round(4))


# ---------------------------------------------------------------------------
# 7. Feature-feature correlation matrix
# ---------------------------------------------------------------------------
section("7. FEATURE-FEATURE CORRELATIONS")
corr_matrix = train_clean[FEATURE_COLS].corr()
high_pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i + 1, len(FEATURE_COLS)):
        c = corr_matrix.iloc[i, j]
        if abs(c) > 0.3:
            high_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], round(c, 4)))
print(f"Pairs with |corr| > 0.3: {len(high_pairs)}")
for pair in sorted(high_pairs, key=lambda x: -abs(x[2]))[:10]:
    print(f"  {pair[0]:>3} -- {pair[1]:<3}  corr = {pair[2]:+.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    square=True, cbar_kws={"shrink": 0.8}, ax=ax,
)
ax.set_title("Feature-feature correlation matrix")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_corr_heatmap.png", bbox_inches="tight")
plt.close()
print(f"\nSaved: {PLOTS_DIR / '03_corr_heatmap.png'}")


# ---------------------------------------------------------------------------
# 8. Quick scatter: top-3 features vs target
# ---------------------------------------------------------------------------
section("8. TOP FEATURES vs TARGET")
top3 = corr_with_target.head(3).index.tolist()
print(f"Top 3 features by |corr| with target: {top3}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(top3):
    axes[i].scatter(train_clean[col], train_clean[TARGET_COL],
                    alpha=0.4, s=10, color="#0053e2")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("target")
    axes[i].set_title(f"{col} vs target (r={corr_with_target[col]:+.3f})")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_top_features_vs_target.png", bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR / '04_top_features_vs_target.png'}")


# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
section("9. SUMMARY")
print(f"""
Train: {len(train)} rows, {len(FEATURE_COLS)} features
Test : {len(test)} rows
Target: range [{y.min():.2f}, {y.max():.2f}], skew = {y.skew():.3f}

Missing: train has {train[FEATURE_COLS].isnull().any(axis=1).sum()} rows with NaN
         test  has {test[FEATURE_COLS].isnull().any(axis=1).sum()} rows with NaN

Top feature correlations with target:
{corr_with_target.head(5).round(4).to_string()}

Plots saved to: {PLOTS_DIR}
""")
