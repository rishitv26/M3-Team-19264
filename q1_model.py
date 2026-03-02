"""
CEX 2020 Expenditure-to-Income Ratio Predictor — UPGRADED
===========================================================
Key improvements over v1:
  1. Actual household income (FINCBTXM / FINATXEM / FINCBTXQ)
  2. Spending ratio as primary target (not raw spending level)
  3. Interaction features (cars_per_person, age², owner×age, income×size)
  4. ElasticNet + XGBoost added to model roster
  5. Hyperparameter-tuned RF (RandomizedSearchCV)
  6. Two-stage model: Stage1=income, Stage2=spending|income
  7. All outputs saved as PNGs + CSVs

Data: https://www.bls.gov/cex/pumd_data.htm  (intrvw20.zip)
Setup: pip install pandas numpy matplotlib scikit-learn xgboost scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob, os, warnings
import sys

_f = open("output.txt", "w")
sys.stdout = _f

warnings.filterwarnings("ignore")



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not installed — skipping XGBRegressor. Run: pip install xgboost")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
DATA_FOLDER = "intrvw20/"

fmli_files = glob.glob(os.path.join(DATA_FOLDER, "**", "[Ff][Mm][Ll][Ii]*.csv"), recursive=True)
print(f"Found {len(fmli_files)} FMLI files:")
for f in sorted(fmli_files):
    n = len(pd.read_csv(f, usecols=["NEWID"]))
    print(f"  {f}  ({n:,} rows)")

if not fmli_files:
    raise FileNotFoundError(f"No FMLI files under '{DATA_FOLDER}'. Check extraction.")

# ── IMPROVEMENT 1: add actual income columns ──────────────────────────────────
# FINCBTXM = income before taxes (monthly CU reference), summed → annual
# FINATXEM = after-tax income estimate
# FINCBTXQ = income before taxes (quarterly)
COLUMNS_NEEDED = [
    "NEWID", "AGE_REF", "REGION", "CUTENURE", "VEHQ",
    "FAM_SIZE", "BLS_URBN",
    "TOTEXPCQ", "HOUSCQ", "TRANSCQ", "FOODCQ", "HEALTHCQ",
    "EDUCCQ", "ENTERTCQ", "FINLWT21",
    "MORT_CQ", "MRPXCQ", "MRGX_CQ", "MRTPXCQ",
    # ▼ NEW: actual income candidates
    "FINCBTXM", "FINATXEM", "FINCBTXQ",
    # ▼ NEW: additional predictors
    "EDUCA2", "EMPLTYP1", "SEX_REF", "MARITAL1",
]

frames = []
for f in sorted(fmli_files):
    df = pd.read_csv(
        f,
        usecols=lambda c: c.upper() in [x.upper() for x in COLUMNS_NEEDED],
        dtype=str,
    )
    df.columns = df.columns.str.upper()
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows loaded: {len(raw):,}")

# Mortgage column resolution
mort_candidates = ["MRPXCQ", "MRGX_CQ", "MRTPXCQ", "MORT_CQ"]
found_mort = [c for c in mort_candidates if c in raw.columns]
raw["MORTGAGE_QTR"] = raw[found_mort[0]] if found_mort else np.nan
for backup in found_mort[1:]:
    raw["MORTGAGE_QTR"] = raw["MORTGAGE_QTR"].fillna(raw[backup])

# Income column resolution
income_candidates = ["FINCBTXM", "FINATXEM", "FINCBTXQ"]
found_income = [c for c in income_candidates if c in raw.columns]
if found_income:
    print(f"  ✓ Income column(s) found: {found_income}")
    raw["INCOME_QTR"] = raw[found_income[0]]
    for backup in found_income[1:]:
        raw["INCOME_QTR"] = raw["INCOME_QTR"].fillna(raw[backup])
    HAS_INCOME = True
else:
    print("  ⚠  No income column found — will use age-group proxy income only")
    raw["INCOME_QTR"] = np.nan
    HAS_INCOME = False

for col in raw.columns:
    if col != "NEWID":
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

# ── 2. AGGREGATE TO ANNUAL HOUSEHOLD LEVEL ────────────────────────────────────
raw["HH_ID"] = raw["NEWID"].astype(str).str[:-1]

EXPENSE_COLS = [c for c in [
    "TOTEXPCQ", "HOUSCQ", "TRANSCQ", "FOODCQ",
    "HEALTHCQ", "EDUCCQ", "ENTERTCQ", "MORTGAGE_QTR", "INCOME_QTR",
] if c in raw.columns]

DEMO_COLS = [c for c in [
    "AGE_REF", "FAM_SIZE", "BLS_URBN", "REGION", "CUTENURE", "VEHQ",
    "FINLWT21", "EDUCA2", "EMPLTYP1", "SEX_REF", "MARITAL1",
] if c in raw.columns]

agg_spec = {col: "first" for col in DEMO_COLS}
agg_spec.update({col: "sum" for col in EXPENSE_COLS})

hh = raw.groupby("HH_ID").agg(agg_spec).reset_index()
print(f"Unique households: {len(hh):,}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
hh["AGE_GROUP"] = pd.cut(
    hh["AGE_REF"],
    bins=[0, 25, 35, 45, 55, 65, 75, 120],
    labels=["<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"],
)

AGE_INCOME_MAP = {
    "<25":   48514,
    "25-34": 102494,
    "35-44": 128285,
    "45-54": 141121,
    "55-64": 121571,
    "65-74": 75460,
    "75+":   56028,
}
hh["AVG_INC_AGE"] = hh["AGE_GROUP"].map(AGE_INCOME_MAP)

hh["REGION_LABEL"]  = hh["REGION"].map({1:"Northeast",2:"Midwest",3:"South",4:"West"})
hh["TENURE_LABEL"]  = hh["CUTENURE"].map({1:"Own w/ Mortgage",2:"Own Free & Clear",3:"Own (unknown)",4:"Rent (no cash)",5:"Renter"})
hh["IS_OWNER"]      = hh["CUTENURE"].isin([1,2,3]).astype(int)
hh["HAS_MORTGAGE"]  = (hh["CUTENURE"] == 1).astype(int)
hh["HAS_CAR"]       = (hh["VEHQ"].fillna(0) > 0).astype(int)
hh["NUM_CARS"]      = hh["VEHQ"].fillna(0)
hh["NUM_RESIDENTS"] = hh["FAM_SIZE"].fillna(1)

# ── IMPROVEMENT 2: resolve best income estimate ───────────────────────────────
if HAS_INCOME:
    # FINCBTXM is a *monthly* income figure recorded once per quarter interview.
    # After groupby-sum across up to 4 quarters, INCOME_QTR ≈ 4 × monthly value.
    # Multiply by 3 to convert: 4 monthly snapshots × 3 = ~12 months = annual.
    hh["INCOME_ANNUAL"] = hh["INCOME_QTR"] * 3
    hh["INCOME_ANNUAL"] = hh["INCOME_ANNUAL"].where(hh["INCOME_ANNUAL"] > 500, hh["AVG_INC_AGE"])
else:
    hh["INCOME_ANNUAL"] = hh["AVG_INC_AGE"]

# Replace zero / negative income with age proxy to avoid division errors
hh["INCOME_ANNUAL"] = hh["INCOME_ANNUAL"].where(hh["INCOME_ANNUAL"] > 500, hh["AVG_INC_AGE"])
hh["INCOME_ANNUAL"] = hh["INCOME_ANNUAL"].fillna(hh["AVG_INC_AGE"])

# ── IMPROVEMENT 3: primary target = spending ratio ────────────────────────────
# Ratio > 1 means household is spending more than income (dis-saving / debt)
hh["SPEND_RATIO"] = hh["TOTEXPCQ"] / hh["INCOME_ANNUAL"]
hh["SPEND_RATIO"]  = hh["SPEND_RATIO"].clip(0.01, 3.0)   # cap extreme outliers
hh["LOG_SPEND"]    = np.log1p(hh["TOTEXPCQ"])
hh["LOG_INCOME"]   = np.log1p(hh["INCOME_ANNUAL"])

# ── IMPROVEMENT 4: interaction / polynomial features ─────────────────────────
hh["CARS_PER_PERSON"]  = hh["NUM_CARS"]      / hh["NUM_RESIDENTS"].replace(0, 1)
hh["AGE_SQUARED"]      = hh["AGE_REF"] ** 2
hh["OWNER_X_AGE"]      = hh["IS_OWNER"]      * hh["AGE_REF"]
hh["INCOME_X_SIZE"]    = hh["INCOME_ANNUAL"] * hh["NUM_RESIDENTS"]
hh["MORTGAGE_FLAG"]    = hh["HAS_MORTGAGE"]  * hh["IS_OWNER"]
hh["LOG_INCOME_X_CAR"] = hh["LOG_INCOME"]    * hh["NUM_CARS"]

print("\n── Demographic Snapshot ──────────────────────────────────────────────")
print(f"  Median age            : {hh['AGE_REF'].median():.0f}")
print(f"  Mean household size   : {hh['NUM_RESIDENTS'].mean():.2f}")
print(f"  Homeownership rate    : {hh['IS_OWNER'].mean()*100:.1f}%")
print(f"  Mean income (annual)  : ${hh['INCOME_ANNUAL'].mean():,.0f}")
print(f"  Median spend ratio    : {hh['SPEND_RATIO'].median():.3f}")

# ── 4. VARIABLE IMPORTANCE vs SPEND_RATIO ────────────────────────────────────
print("\n── Spearman Importance vs Spending Ratio ────────────────────────────")
target = "SPEND_RATIO"
exclude = {
    "SPEND_RATIO", "TOTEXPCQ", "HOUSCQ", "TRANSCQ", "FOODCQ",
    "HEALTHCQ", "EDUCCQ", "ENTERTCQ", "MORTGAGE_QTR", "INCOME_QTR",
    "LOG_SPEND", "HH_ID", "NEWID", "AGE_GROUP", "REGION_LABEL", "TENURE_LABEL",
}
candidate_cols = [c for c in hh.columns if c not in exclude]

importance_rows = []
for col in candidate_cols:
    # Skip non-numeric / categorical columns
    if not pd.api.types.is_numeric_dtype(hh[col]):
        continue
    sub = hh[[col, target]].dropna()
    sub = sub[(sub[target] > 0)]
    try:
        sub = sub[np.isfinite(sub[col]) & np.isfinite(sub[target])]
    except TypeError:
        continue
    if len(sub) < 50 or sub[col].nunique() < 2:
        continue
    rho, pval = spearmanr(sub[col], sub[target])
    try:
        bins = pd.qcut(sub[col], q=10, duplicates="drop")
        gm   = sub.groupby(bins, observed=True)[target].mean()
        om   = sub[target].mean()
        ss_b = sum((sub.groupby(bins, observed=True)[target].count() * (gm - om)**2).dropna())
        ss_t = ((sub[target] - om)**2).sum()
        eta  = ss_b / ss_t if ss_t > 0 else 0
    except Exception:
        eta = np.nan
    importance_rows.append({"Column":col,"Spearman_rho":round(rho,4),"Abs_rho":round(abs(rho),4),
                             "p_value":round(pval,6),"Eta_squared":round(eta,4) if not np.isnan(eta) else np.nan,"N":len(sub)})

imp_df = pd.DataFrame(importance_rows).sort_values("Abs_rho", ascending=False)
print(f"\n  {'Column':<22} {'ρ':>10} {'|ρ|':>8} {'Eta²':>8} {'N':>7}")
print("  " + "-" * 62)
for _, row in imp_df.iterrows():
    d = "↑" if row["Spearman_rho"] > 0 else "↓"
    print(f"  {row['Column']:<22} {row['Spearman_rho']:>+9.4f}{d}  "
          f"{row['Abs_rho']:>7.4f}  "
          f"{str(round(row['Eta_squared'],4)) if pd.notna(row['Eta_squared']) else 'N/A':>8}  "
          f"{row['N']:>6,}")

# ── 5. MODEL ROSTER ───────────────────────────────────────────────────────────
print("\n── Building Model Roster ────────────────────────────────────────────")

# Feature selection: top features by combined RF+Spearman rank
LEAKY  = {"TOTEXPCQ","HOUSCQ","TRANSCQ","FOODCQ","HEALTHCQ","EDUCCQ","ENTERTCQ",
           "MORTGAGE_QTR","INCOME_QTR","LOG_SPEND","SPEND_RATIO","AVG_INC_AGE",
           "INCOME_ANNUAL"}  # income feeds ratio; keep log version as feature
LABELS = {"HH_ID","AGE_GROUP","REGION_LABEL","TENURE_LABEL"}

feat_pool = [c for c in imp_df["Column"].tolist()
             if c not in LEAKY and c not in LABELS and c in hh.columns]

# Quick RF scan to find optimal N features
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scan_results = []
for n in range(1, min(len(feat_pool)+1, 25)):
    feats = feat_pool[:n]
    sub   = hh[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    sub   = sub[(sub[target] > 0)]
    if len(sub) < 100:
        continue
    r2 = cross_val_score(
        RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        sub[feats].values, sub[target].values, cv=kf, scoring="r2"
    ).mean()
    scan_results.append({"n_features":n,"R2":r2,"features":feats})
    print(f"    n={n:>2}  R2={r2:.4f}  [{', '.join(feats)}]")

scan_df    = pd.DataFrame(scan_results)
best_n     = int(scan_df.loc[scan_df["R2"].idxmax(), "n_features"])
FINAL_FEATS= feat_pool[:best_n]
print(f"\n  Optimal: {best_n} features → {FINAL_FEATS}  CV R2={scan_df['R2'].max():.4f}")

ml_df = hh[FINAL_FEATS + [target]].replace([np.inf,-np.inf], np.nan).dropna()
ml_df = ml_df[ml_df[target] > 0]
X_fin = ml_df[FINAL_FEATS].values
y_fin = ml_df[target].values   # predict ratio directly (not log, already bounded)

# ── IMPROVEMENT 5: expanded model roster ─────────────────────────────────────
MODELS = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge Regression":  Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "ElasticNet":        Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))]),
    "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
}
if HAS_XGB:
    MODELS["XGBoost"] = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )

model_results = {}
print(f"\n  {'Model':<22} {'R2':>8} {'std':>7} {'RMSE':>10}")
print("  " + "-" * 52)
for name, model in MODELS.items():
    r2s  = cross_val_score(model, X_fin, y_fin, cv=kf, scoring="r2")
    rmse = np.sqrt(-cross_val_score(model, X_fin, y_fin, cv=kf, scoring="neg_mean_squared_error"))
    model_results[name] = {"R2": r2s.mean(), "R2_std": r2s.std(), "RMSE": rmse.mean()}
    print(f"  {name:<22} {r2s.mean():>8.4f} {r2s.std():>7.4f} {rmse.mean():>10.4f}")

best_name  = max(model_results, key=lambda k: model_results[k]["R2"])
best_model = MODELS[best_name]

# ── IMPROVEMENT 6: hyperparameter tuning on best tree model ──────────────────
print(f"\n── Tuning hyperparameters for best model: {best_name} ──────────────")
if "Forest" in best_name:
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth":     [None, 6, 8, 12],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
    }
    tuner = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist, n_iter=20, cv=kf, scoring="r2", random_state=42, n_jobs=-1
    )
    tuner.fit(X_fin, y_fin)
    tuned_r2 = tuner.best_score_
    best_model = tuner.best_estimator_
    print(f"  Tuned R2: {tuned_r2:.4f}  params: {tuner.best_params_}")
    model_results[best_name + " (tuned)"] = {"R2": tuned_r2, "R2_std": 0, "RMSE": 0}
elif "Boosting" in best_name or "XGB" in best_name:
    best_model.fit(X_fin, y_fin)  # keep defaults, already reasonable
    print("  Using default params (already tuned heuristically).")
else:
    best_model.fit(X_fin, y_fin)

best_model.fit(X_fin, y_fin)
print(f"\n  Best model: {best_name}  R2={model_results[best_name]['R2']:.4f}")

# ── 7. TWO-STAGE MODEL ────────────────────────────────────────────────────────
print("\n── Two-Stage Model: Stage1=Income, Stage2=Spending|Income ──────────")

# Stage 1: predict log(income) from pure demographics (no income features)
stage1_feats_pool = [
    "AGE_REF", "AGE_SQUARED", "NUM_RESIDENTS", "IS_OWNER", "HAS_MORTGAGE",
    "NUM_CARS", "CARS_PER_PERSON", "OWNER_X_AGE", "BLS_URBN",
    "REGION", "CUTENURE", "EDUCA2", "EMPLTYP1", "SEX_REF", "MARITAL1",
]
s1_feats  = [c for c in stage1_feats_pool if c in hh.columns]
s1_df     = hh[s1_feats + ["INCOME_ANNUAL"]].replace([np.inf,-np.inf], np.nan).dropna()
s1_df     = s1_df[s1_df["INCOME_ANNUAL"] > 500]
X_s1      = s1_df[s1_feats].values
y_s1      = np.log1p(s1_df["INCOME_ANNUAL"].values)

s1_model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
s1_r2    = cross_val_score(s1_model, X_s1, y_s1, cv=kf, scoring="r2")
print(f"  Stage 1  Predict log(income)  CV R2 = {s1_r2.mean():.4f} ± {s1_r2.std():.4f}")
s1_model.fit(X_s1, y_s1)

# Attach predicted income directly via the s1_df index (no merge needed)
# s1_df was filtered from hh, so its index IS the hh index — use .loc directly
hh["PRED_LOG_INC"] = np.nan
hh.loc[s1_df.index, "PRED_LOG_INC"] = s1_model.predict(X_s1)
hh["PRED_LOG_INC"] = hh["PRED_LOG_INC"].fillna(np.log1p(hh["AVG_INC_AGE"].astype(float)))  # fallback for rows outside s1

# Stage 2: predict spend ratio using predicted income — no actual income leakage
# Crucially, INCOME_ANNUAL / LOG_INCOME must NOT be features here (they come from actual income)
s2_base_feats = [c for c in s1_feats if c in hh.columns]  # pure demographics
s2_feats      = s2_base_feats + ["PRED_LOG_INC"]           # + predicted income only

s2_df  = hh[s2_feats + ["SPEND_RATIO"]].replace([np.inf, -np.inf], np.nan).dropna()
s2_df  = s2_df[s2_df["SPEND_RATIO"] > 0]
X_s2   = s2_df[s2_feats].values
y_s2   = s2_df["SPEND_RATIO"].values

s2_model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
s2_r2    = cross_val_score(s2_model, X_s2, y_s2, cv=kf, scoring="r2")
print(f"  Stage 2  Predict spend ratio  CV R2 = {s2_r2.mean():.4f} ± {s2_r2.std():.4f}")
s2_model.fit(X_s2, y_s2)

baseline_r2 = model_results.get("Gradient Boosting", {}).get("R2", 0)
print(f"\n  Two-stage R2 vs single-stage baseline: {s2_r2.mean():.4f} vs {baseline_r2:.4f}"
      f"  ({'↑ improvement' if s2_r2.mean() > baseline_r2 else '↓ no gain — income proxy is the bottleneck'})")

# ── 8. VISUALISATIONS ─────────────────────────────────────────────────────────
PALETTE = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4","#FF5722"]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("CEX 2020 — Upgraded: Spending Ratio Analysis", fontsize=15, fontweight="bold", y=1.01)

# 8a. Spend ratio by age group
ax = axes[0, 0]
age_ratio = hh.dropna(subset=["AGE_GROUP"]).groupby("AGE_GROUP", observed=True)["SPEND_RATIO"].median()
age_ratio.plot(kind="bar", ax=ax, color=PALETTE[:len(age_ratio)], edgecolor="white")
ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Ratio = 1.0")
ax.set_title("Median Spend Ratio by Age Group")
ax.set_xlabel("Age group"); ax.set_ylabel("Spend / Income")
ax.legend(); ax.tick_params(axis="x", rotation=30)

# 8b. Spend ratio by tenure
ax = axes[0, 1]
ten_ratio = (hh.dropna(subset=["TENURE_LABEL"])
             .groupby("TENURE_LABEL")["SPEND_RATIO"].median()
             .sort_values(ascending=False))
ten_ratio.plot(kind="bar", ax=ax, color=PALETTE[:len(ten_ratio)], edgecolor="white")
ax.axhline(1.0, color="red", linestyle="--", alpha=0.7)
ax.set_title("Median Spend Ratio by Tenure"); ax.set_xlabel("")
ax.set_ylabel("Spend / Income"); ax.tick_params(axis="x", rotation=20)

# 8c. Model R2 comparison
ax = axes[0, 2]
disp_results = {k:v for k,v in model_results.items() if v["R2"] != 0}
names  = list(disp_results.keys())
r2vals = [disp_results[n]["R2"] for n in names]
colors = ["#4CAF50" if n == best_name or "(tuned)" in n else "#2196F3" for n in names]
bars   = ax.bar(names, r2vals, color=colors, edgecolor="white")
ax.set_title("CV R2 by Model (Spend Ratio Target)")
ax.set_ylabel("R2"); ax.set_ylim(0, max(r2vals)*1.25)
ax.tick_params(axis="x", rotation=25)
for bar, val in zip(bars, r2vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)

# 8d. Feature importance (best model)
ax = axes[1, 0]
if hasattr(best_model, "feature_importances_"):
    fi = best_model.feature_importances_
    pd.Series(fi, index=FINAL_FEATS).sort_values().plot(
        kind="barh", ax=ax, color="#9C27B0", edgecolor="white")
    ax.set_title(f"Feature Importance\n({best_name})")
    ax.set_xlabel("Importance")
else:
    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

# 8e. R2 scan curve
ax = axes[1, 1]
ax.plot(scan_df["n_features"], scan_df["R2"], marker="o", color="#2196F3", linewidth=2)
ax.axvline(best_n, color="#E91E63", linestyle="--", label=f"Optimal n={best_n}")
ax.set_xlabel("Number of features"); ax.set_ylabel("CV R2")
ax.set_title("R2 vs Feature Count"); ax.legend(); ax.grid(True, alpha=0.3)

# 8f. Distribution of spend ratio
ax = axes[1, 2]
ratio = hh["SPEND_RATIO"].dropna()
ratio = ratio[(ratio > 0) & (ratio < 3)]
ax.hist(ratio, bins=60, color="#FF9800", edgecolor="white", alpha=0.85)
ax.axvline(1.0, color="red", linestyle="--", label="Ratio = 1.0")
ax.set_title("Distribution of Spending Ratio\n(Spending / Income)")
ax.set_xlabel("Spend / Income Ratio"); ax.set_ylabel("Households"); ax.legend()

plt.tight_layout()
plt.savefig("cex_2020_upgraded_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: cex_2020_upgraded_overview.png")

# ── 9. SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  UPGRADE SUMMARY")
print("="*65)
print(f"  Income source       : {'Actual FMLI income' if HAS_INCOME else 'Age-group proxy'}")
print(f"  Primary target      : SPEND_RATIO (spending / income)")
print(f"  Features used       : {best_n}  →  {FINAL_FEATS}")
print(f"  Best single model   : {best_name}  R2={model_results[best_name]['R2']:.4f}")
print(f"  Two-stage S2 R2     : {s2_r2.mean():.4f}")
print(f"  Stage 1 income R2   : {s1_r2.mean():.4f}")
print("="*65)

imp_df.to_csv("variable_importance_ratio.csv", index=False)
print("  ✓ Saved: variable_importance_ratio.csv")

sys.stdout = sys.__stdout__  # restore
_f.close()
