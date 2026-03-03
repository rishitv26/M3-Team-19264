"""
Q1: Playing With House Money

"""

from __future__ import annotations

import glob
import os
import pickle
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False                                                             

PREDICT_FEATURES = [
    "LOG_INCOME", "INCOME_X_SIZE", "LOG_INCOME_X_CAR",
    "CUTENURE", "IS_OWNER", "MORTGAGE_FLAG", "HAS_MORTGAGE", "NUM_CARS",
]

AGE_INCOME_MAP = {
    "<25": 48514, "25-34": 102494, "35-44": 128285,
    "45-54": 141121, "55-64": 121571, "65-74": 75460, "75+": 56028,
}

LEAKY = {
    "TOTEXPCQ", "HOUSCQ", "TRANSCQ", "FOODCQ", "HEALTHCQ",
    "EDUCCQ", "ENTERTCQ", "MORTGAGE_QTR", "INCOME_QTR",
    "LOG_SPEND", "SPEND_RATIO", "AVG_INC_AGE", "INCOME_ANNUAL",
}
LABELS = {"HH_ID", "AGE_GROUP", "REGION_LABEL", "TENURE_LABEL"}
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]


class SpendingRatioPredictor:
    """Full experimental pipeline + prediction interface for CEX 2020 spending ratio."""

    def __init__(
        self,
        data_folder:  str = "intrvw20/",
        cv_folds:     int = 5,
        random_state: int = 42,
    ):
        self.data_folder  = data_folder
        self.cv_folds     = cv_folds
        self.random_state = random_state

        self.hh:           Optional[pd.DataFrame] = None   # load_data()
        self.imp_df:       Optional[pd.DataFrame] = None   # run_importance()
        self.scan_df:      Optional[pd.DataFrame] = None   # run_feature_scan()
        self.best_n:       Optional[int]           = None
        self.final_feats:  Optional[list]          = None

        self.model_results:  dict = {}   # name  -> {R2, R2_std, RMSE}
        self.trained_models: dict = {}   # name  -> fitted model object
        self.best_model_name: Optional[str] = None

        self.tuned_model:  Optional[object] = None
        self.tuned_r2:     Optional[float]  = None
        self.tuned_params: Optional[dict]   = None

        self.s1_model: Optional[object] = None
        self.s2_model: Optional[object] = None
        self.s1_r2:    Optional[float]  = None
        self.s2_r2:    Optional[float]  = None
        self.s1_feats: Optional[list]   = None
        self.s2_feats: Optional[list]   = None

        # Active model used by predict()
        self._active_model: Optional[object] = None
        self._active_feats: Optional[list]   = None
        self._active_name:  Optional[str]    = None

    # STEP 1 — Load & engineer
      

    def load_data(self, verbose: bool = True) -> "SpendingRatioPredictor":
        """Load FMLI CSVs, aggregate to household level, engineer all features."""
        fmli_files = glob.glob(
            os.path.join(self.data_folder, "**", "[Ff][Mm][Ll][Ii]*.csv"),
            recursive=True,
        )
        if not fmli_files:
            raise FileNotFoundError(f"No FMLI files found under '{self.data_folder}'.")

        if verbose:
            print(f"Found {len(fmli_files)} FMLI files:")
            for f in sorted(fmli_files):
                n = len(pd.read_csv(f, usecols=["NEWID"]))
                print(f"  {f}  ({n:,} rows)")

        COLUMNS_NEEDED = [
            "NEWID", "AGE_REF", "REGION", "CUTENURE", "VEHQ", "FAM_SIZE", "BLS_URBN",
            "TOTEXPCQ", "HOUSCQ", "TRANSCQ", "FOODCQ", "HEALTHCQ",
            "EDUCCQ", "ENTERTCQ", "FINLWT21",
            "MORT_CQ", "MRPXCQ", "MRGX_CQ", "MRTPXCQ",
            "FINCBTXM", "FINATXEM", "FINCBTXQ",
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
        if verbose:
            print(f"\nTotal rows loaded: {len(raw):,}")

        # Mortgage
        mort_candidates = ["MRPXCQ", "MRGX_CQ", "MRTPXCQ", "MORT_CQ"]
        found_mort = [c for c in mort_candidates if c in raw.columns]
        raw["MORTGAGE_QTR"] = raw[found_mort[0]] if found_mort else np.nan
        for backup in found_mort[1:]:
            raw["MORTGAGE_QTR"] = raw["MORTGAGE_QTR"].fillna(raw[backup])

        # Income
        income_candidates = ["FINCBTXM", "FINATXEM", "FINCBTXQ"]
        found_income = [c for c in income_candidates if c in raw.columns]
        if found_income:
            if verbose:
                print(f"  ✓ Income column(s) found: {found_income}")
            raw["INCOME_QTR"] = raw[found_income[0]]
            for backup in found_income[1:]:
                raw["INCOME_QTR"] = raw["INCOME_QTR"].fillna(raw[backup])
            has_income = True
        else:
            raw["INCOME_QTR"] = np.nan
            has_income = False

        for col in raw.columns:
            if col != "NEWID":
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

        raw["HH_ID"] = raw["NEWID"].astype(str).str[:-1]

        EXPENSE_COLS = [c for c in [
            "TOTEXPCQ", "HOUSCQ", "TRANSCQ", "FOODCQ", "HEALTHCQ",
            "EDUCCQ", "ENTERTCQ", "MORTGAGE_QTR", "INCOME_QTR",
        ] if c in raw.columns]
        DEMO_COLS = [c for c in [
            "AGE_REF", "FAM_SIZE", "BLS_URBN", "REGION", "CUTENURE", "VEHQ",
            "FINLWT21", "EDUCA2", "EMPLTYP1", "SEX_REF", "MARITAL1",
        ] if c in raw.columns]

        agg_spec = {col: "first" for col in DEMO_COLS}
        agg_spec.update({col: "sum" for col in EXPENSE_COLS})
        hh = raw.groupby("HH_ID").agg(agg_spec).reset_index()

        if verbose:
            print(f"Unique households: {len(hh):,}")

        # Feature engineering
        hh["AGE_GROUP"] = pd.cut(
            hh["AGE_REF"],
            bins=[0, 25, 35, 45, 55, 65, 75, 120],
            labels=list(AGE_INCOME_MAP.keys()),
        )
        hh["AVG_INC_AGE"]   = hh["AGE_GROUP"].map(AGE_INCOME_MAP).astype(float)
        hh["REGION_LABEL"]  = hh["REGION"].map({1: "Northeast", 2: "Midwest", 3: "South", 4: "West"})
        hh["TENURE_LABEL"]  = hh["CUTENURE"].map({
            1: "Own w/ Mortgage", 2: "Own Free & Clear",
            3: "Own (unknown)", 4: "Rent (no cash)", 5: "Renter",
        })
        hh["IS_OWNER"]      = hh["CUTENURE"].isin([1, 2, 3]).astype(int)
        hh["HAS_MORTGAGE"]  = (hh["CUTENURE"] == 1).astype(int)
        hh["HAS_CAR"]       = (hh["VEHQ"].fillna(0) > 0).astype(int)
        hh["NUM_CARS"]      = hh["VEHQ"].fillna(0)
        hh["NUM_RESIDENTS"] = hh["FAM_SIZE"].fillna(1)

        if has_income:
            hh["INCOME_ANNUAL"] = hh["INCOME_QTR"] * 3
            hh["INCOME_ANNUAL"] = hh["INCOME_ANNUAL"].where(hh["INCOME_ANNUAL"] > 500, hh["AVG_INC_AGE"])
        else:
            hh["INCOME_ANNUAL"] = hh["AVG_INC_AGE"]
        hh["INCOME_ANNUAL"] = (
            hh["INCOME_ANNUAL"]
            .where(hh["INCOME_ANNUAL"] > 500, hh["AVG_INC_AGE"])
            .fillna(hh["AVG_INC_AGE"])
        )

        hh["SPEND_RATIO"]      = (hh["TOTEXPCQ"] / hh["INCOME_ANNUAL"]).clip(0.01, 3.0)
        hh["LOG_SPEND"]        = np.log1p(hh["TOTEXPCQ"])
        hh["LOG_INCOME"]       = np.log1p(hh["INCOME_ANNUAL"])
        hh["CARS_PER_PERSON"]  = hh["NUM_CARS"] / hh["NUM_RESIDENTS"].replace(0, 1)
        hh["AGE_SQUARED"]      = hh["AGE_REF"] ** 2
        hh["OWNER_X_AGE"]      = hh["IS_OWNER"] * hh["AGE_REF"]
        hh["INCOME_X_SIZE"]    = hh["INCOME_ANNUAL"] * hh["NUM_RESIDENTS"]
        hh["MORTGAGE_FLAG"]    = hh["HAS_MORTGAGE"] * hh["IS_OWNER"]
        hh["LOG_INCOME_X_CAR"] = hh["LOG_INCOME"] * hh["NUM_CARS"]

        if verbose:
            print(f"\n   Demographic Snapshot                                            ")
            print(f"  Median age            : {hh['AGE_REF'].median():.0f}")
            print(f"  Mean household size   : {hh['NUM_RESIDENTS'].mean():.2f}")
            print(f"  Homeownership rate    : {hh['IS_OWNER'].mean()*100:.1f}%")
            print(f"  Mean income (annual)  : ${hh['INCOME_ANNUAL'].mean():,.0f}")
            print(f"  Median spend ratio    : {hh['SPEND_RATIO'].median():.3f}")

        self.hh = hh
        return self

      
    # STEP 2 — Variable importance
      

    def run_importance(self, verbose: bool = True) -> "SpendingRatioPredictor":
        """Compute Spearman ρ and Eta² for all candidate features vs SPEND_RATIO."""
        self._require("hh")
        hh, target = self.hh, "SPEND_RATIO"
        exclude = LEAKY | LABELS | {target, "HH_ID", "NEWID"}

        rows = []
        for col in [c for c in hh.columns if c not in exclude]:
            if not pd.api.types.is_numeric_dtype(hh[col]):
                continue
            sub = hh[[col, target]].dropna()
            sub = sub[sub[target] > 0]
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
                ss_b = sum((sub.groupby(bins, observed=True)[target].count() * (gm - om) ** 2).dropna())
                ss_t = ((sub[target] - om) ** 2).sum()
                eta  = ss_b / ss_t if ss_t > 0 else 0
            except Exception:
                eta = np.nan
            rows.append({
                "Column": col, "Spearman_rho": round(rho, 4),
                "Abs_rho": round(abs(rho), 4), "p_value": round(pval, 6),
                "Eta_squared": round(eta, 4) if not np.isnan(eta) else np.nan,
                "N": len(sub),
            })

        self.imp_df = pd.DataFrame(rows).sort_values("Abs_rho", ascending=False)

        if verbose:
            print("\n   Spearman Importance vs Spending Ratio                             ")
            print(f"\n  {'Column':<22} {'ρ':>10} {'|ρ|':>8} {'Eta²':>8} {'N':>7}")
            print("  " + "-" * 62)
            for _, row in self.imp_df.iterrows():
                d = "↑" if row["Spearman_rho"] > 0 else "↓"
                print(f"  {row['Column']:<22} {row['Spearman_rho']:>+9.4f}{d}  "
                      f"{row['Abs_rho']:>7.4f}  "
                      f"{str(round(row['Eta_squared'], 4)) if pd.notna(row['Eta_squared']) else 'N/A':>8}  "
                      f"{row['N']:>6,}")

        self.imp_df.to_csv("variable_importance_ratio.csv", index=False)
        return self
      

    def run_feature_scan(self, max_feats: int = 25, verbose: bool = True) -> "SpendingRatioPredictor":
        """Scan 1 ->N features (by importance rank) to find the optimal count via CV R²."""
        self._require("imp_df")
        hh, target = self.hh, "SPEND_RATIO"
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        feat_pool = [
            c for c in self.imp_df["Column"].tolist()
            if c not in LEAKY and c not in LABELS and c in hh.columns
        ]

        if verbose:
            print("\n   Feature Count Scan (RF CV R²)                                     ")

        results = []
        for n in range(1, min(len(feat_pool) + 1, max_feats + 1)):
            feats = feat_pool[:n]
            sub   = hh[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
            sub   = sub[sub[target] > 0]
            if len(sub) < 100:
                continue
            r2 = cross_val_score(
                RandomForestRegressor(n_estimators=100, max_depth=8,
                                      random_state=self.random_state, n_jobs=-1),
                sub[feats].values, sub[target].values, cv=kf, scoring="r2",
            ).mean()
            results.append({"n_features": n, "R2": r2, "features": feats})
            if verbose:
                print(f"    n={n:>2}  R2={r2:.4f}  [{', '.join(feats)}]")

        self.scan_df    = pd.DataFrame(results)
        self.best_n     = int(self.scan_df.loc[self.scan_df["R2"].idxmax(), "n_features"])
        self.final_feats = feat_pool[:self.best_n]

        if verbose:
            print(f"\n  Optimal: {self.best_n} features  -> {self.final_feats}  "
                  f"CV R²={self.scan_df['R2'].max():.4f}")
        return self

      
    # STEP 4 — Train all models
      

    def run_all_models(self, verbose: bool = True) -> "SpendingRatioPredictor":
        """Train and cross-validate all models on the optimal feature set."""
        self._require("final_feats")
        hh, target = self.hh, "SPEND_RATIO"
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        ml_df = hh[self.final_feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
        ml_df = ml_df[ml_df[target] > 0]
        X, y  = ml_df[self.final_feats].values, ml_df[target].values

        MODELS = {
            "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "Ridge Regression":  Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            "ElasticNet":        Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))]),
            "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=8, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=self.random_state),
        }
        if HAS_XGB:
            from xgboost import XGBRegressor
            MODELS["XGBoost"] = XGBRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1, verbosity=0,
            )

        if verbose:
            print(f"\n   Model Comparison                                                  ")
            print(f"  Features: {self.final_feats}")
            print(f"\n  {'Model':<22} {'R2':>8} {'std':>7} {'RMSE':>10}")
            print("  " + "-" * 52)

        for name, model in MODELS.items():
            r2s  = cross_val_score(model, X, y, cv=kf, scoring="r2")
            rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
            self.model_results[name] = {"R2": r2s.mean(), "R2_std": r2s.std(), "RMSE": rmse.mean()}
            model.fit(X, y)
            self.trained_models[name] = model
            if verbose:
                print(f"  {name:<22} {r2s.mean():>8.4f} {r2s.std():>7.4f} {rmse.mean():>10.4f}")

        self.best_model_name = max(self.model_results, key=lambda k: self.model_results[k]["R2"])

        # Default active model for predict()
        self._active_model = self.trained_models[self.best_model_name]
        self._active_feats = self.final_feats
        self._active_name  = self.best_model_name

        if verbose:
            print(f"\n  Best model: {self.best_model_name}  "
                  f"R²={self.model_results[self.best_model_name]['R2']:.4f}")
        return self

      
    # STEP 5 — Hyperparameter tuning
      

    def tune_best(self, verbose: bool = True) -> "SpendingRatioPredictor":
        """Tune the best model from run_all_models() via RandomizedSearchCV."""
        self._require("best_model_name")
        hh, target = self.hh, "SPEND_RATIO"
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        ml_df = hh[self.final_feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
        ml_df = ml_df[ml_df[target] > 0]
        X, y  = ml_df[self.final_feats].values, ml_df[target].values

        name = self.best_model_name
        if verbose:
            print(f"\n   Tuning: {name}                                                     ")

        if "Forest" in name:
            param_dist = {
                "n_estimators":     [200, 300, 500],
                "max_depth":        [None, 6, 8, 12],
                "min_samples_leaf": [1, 2, 5],
                "max_features":     ["sqrt", "log2", 0.5],
            }
            base = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        elif "Boosting" in name:
            param_dist = {
                "n_estimators":  [200, 300, 500],
                "max_depth":     [3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample":     [0.7, 0.8, 1.0],
            }
            base = GradientBoostingRegressor(random_state=self.random_state)
        elif "XGB" in name and HAS_XGB:
            from xgboost import XGBRegressor
            param_dist = {
                "n_estimators":  [300, 400, 500],
                "max_depth":     [4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample":     [0.7, 0.8, 1.0],
            }
            base = XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0)
        else:
            if verbose:
                print(f"  No tuning grid defined for '{name}' — skipping.")
            return self

        tuner = RandomizedSearchCV(
            base, param_dist, n_iter=20, cv=kf,
            scoring="r2", random_state=self.random_state, n_jobs=-1,
        )
        tuner.fit(X, y)
        self.tuned_model  = tuner.best_estimator_
        self.tuned_r2     = tuner.best_score_
        self.tuned_params = tuner.best_params_

        tuned_name = name + " (tuned)"
        self.model_results[tuned_name]  = {"R2": self.tuned_r2, "R2_std": 0, "RMSE": 0}
        self.trained_models[tuned_name] = self.tuned_model

        # Update active model to tuned version
        self._active_model = self.tuned_model
        self._active_feats = self.final_feats
        self._active_name  = tuned_name

        if verbose:
            print(f"  Tuned R²: {self.tuned_r2:.4f}  params: {self.tuned_params}")
        return self

      
    # STEP 6 — Two-stage model
      

    def run_two_stage(self, verbose: bool = True) -> "SpendingRatioPredictor":
        """
        Stage 1: predict log(income) from pure demographics.
        Stage 2: predict spend ratio using predicted income — no actual income leakage.
        Stores s1_model and s2_model; use swap_model('two_stage') to predict with it.
        """
        self._require("hh")
        hh, target = self.hh, "SPEND_RATIO"
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        if verbose:
            print("\n   Two-Stage Model                                                   ")

        stage1_pool = [
            "AGE_REF", "AGE_SQUARED", "NUM_RESIDENTS", "IS_OWNER", "HAS_MORTGAGE",
            "NUM_CARS", "CARS_PER_PERSON", "OWNER_X_AGE", "BLS_URBN",
            "REGION", "CUTENURE", "EDUCA2", "EMPLTYP1", "SEX_REF", "MARITAL1",
        ]
        self.s1_feats = [c for c in stage1_pool if c in hh.columns]

        s1_df = hh[self.s1_feats + ["INCOME_ANNUAL"]].replace([np.inf, -np.inf], np.nan).dropna()
        s1_df = s1_df[s1_df["INCOME_ANNUAL"] > 500]
        X_s1  = s1_df[self.s1_feats].values
        y_s1  = np.log1p(s1_df["INCOME_ANNUAL"].values)

        self.s1_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=self.random_state,
        )
        s1_scores  = cross_val_score(self.s1_model, X_s1, y_s1, cv=kf, scoring="r2")
        self.s1_r2 = s1_scores.mean()
        self.s1_model.fit(X_s1, y_s1)

        if verbose:
            print(f"  Stage 1  Predict log(income)  CV R² = {self.s1_r2:.4f} ± {s1_scores.std():.4f}")

        hh["PRED_LOG_INC"] = np.nan
        hh.loc[s1_df.index, "PRED_LOG_INC"] = self.s1_model.predict(X_s1)
        hh["PRED_LOG_INC"] = hh["PRED_LOG_INC"].fillna(np.log1p(hh["AVG_INC_AGE"].astype(float)))

        self.s2_feats = self.s1_feats + ["PRED_LOG_INC"]
        s2_df = hh[self.s2_feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
        s2_df = s2_df[s2_df[target] > 0]
        X_s2  = s2_df[self.s2_feats].values
        y_s2  = s2_df[target].values

        self.s2_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=self.random_state,
        )
        s2_scores  = cross_val_score(self.s2_model, X_s2, y_s2, cv=kf, scoring="r2")
        self.s2_r2 = s2_scores.mean()
        self.s2_model.fit(X_s2, y_s2)

        # Register two-stage as a swappable model option
        self.trained_models["two_stage"] = self.s2_model
        self.model_results["two_stage"]  = {"R2": self.s2_r2, "R2_std": s2_scores.std(), "RMSE": 0}

        baseline = self.model_results.get("Gradient Boosting", {}).get("R2", 0)
        arrow    = "↑ improvement" if self.s2_r2 > baseline else "↓ no gain vs single-stage"

        if verbose:
            print(f"  Stage 2  Predict spend ratio  CV R² = {self.s2_r2:.4f} ± {s2_scores.std():.4f}")
            print(f"  Two-stage vs single-stage: {self.s2_r2:.4f} vs {baseline:.4f}  ({arrow})")

        return self

      
    # STEP 7 — Plots
      

    def plot(self, save: bool = True) -> "SpendingRatioPredictor":
        """Generate and save all visualisation panels."""
        self._require("hh")
        hh = self.hh

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("CEX 2020 — Spending Ratio Analysis", fontsize=15, fontweight="bold", y=1.01)

        ax = axes[0, 0]
        age_ratio = hh.dropna(subset=["AGE_GROUP"]).groupby("AGE_GROUP", observed=True)["SPEND_RATIO"].median()
        age_ratio.plot(kind="bar", ax=ax, color=PALETTE[:len(age_ratio)], edgecolor="white")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Ratio=1.0")
        ax.set_title("Median Spend Ratio by Age Group")
        ax.set_xlabel("Age group"); ax.set_ylabel("Spend / Income")
        ax.legend(); ax.tick_params(axis="x", rotation=30)

        ax = axes[0, 1]
        ten_ratio = (hh.dropna(subset=["TENURE_LABEL"])
                     .groupby("TENURE_LABEL")["SPEND_RATIO"].median()
                     .sort_values(ascending=False))
        ten_ratio.plot(kind="bar", ax=ax, color=PALETTE[:len(ten_ratio)], edgecolor="white")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Median Spend Ratio by Tenure")
        ax.set_ylabel("Spend / Income"); ax.tick_params(axis="x", rotation=20)

        ax = axes[0, 2]
        if self.model_results:
            disp   = {k: v for k, v in self.model_results.items() if v["R2"] != 0}
            names  = list(disp.keys())
            r2vals = [disp[n]["R2"] for n in names]
            colors = ["#4CAF50" if n == self._active_name else "#2196F3" for n in names]
            bars   = ax.bar(names, r2vals, color=colors, edgecolor="white")
            ax.set_title("CV R² by Model"); ax.set_ylabel("R²")
            ax.set_ylim(0, max(r2vals) * 1.25); ax.tick_params(axis="x", rotation=25)
            for bar, val in zip(bars, r2vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        ax = axes[1, 0]
        if self._active_model and hasattr(self._active_model, "feature_importances_"):
            pd.Series(self._active_model.feature_importances_, index=self._active_feats).sort_values().plot(
                kind="barh", ax=ax, color="#9C27B0", edgecolor="white")
            ax.set_title(f"Feature Importance\n({self._active_name})")
            ax.set_xlabel("Importance")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 1]
        if self.scan_df is not None:
            ax.plot(self.scan_df["n_features"], self.scan_df["R2"], marker="o", color="#2196F3", linewidth=2)
            ax.axvline(self.best_n, color="#E91E63", linestyle="--", label=f"Optimal n={self.best_n}")
            ax.set_xlabel("Number of features"); ax.set_ylabel("CV R²")
            ax.set_title("R² vs Feature Count"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ratio = hh["SPEND_RATIO"].dropna()
        ratio = ratio[(ratio > 0) & (ratio < 3)]
        ax.hist(ratio, bins=60, color="#FF9800", edgecolor="white", alpha=0.85)
        ax.axvline(1.0, color="red", linestyle="--", label="Ratio=1.0")
        ax.set_title("Distribution of Spending Ratio")
        ax.set_xlabel("Spend / Income"); ax.set_ylabel("Households"); ax.legend()

        plt.tight_layout()
        if save:
            plt.savefig("cex_2020_results.png", dpi=150, bbox_inches="tight")
            print("Saved cex_2020_results.png")
        plt.show()
        return self

      
    # Prediction interface
      

    def predict(
        self,
        LOG_INCOME:       float,
        INCOME_X_SIZE:    float,
        LOG_INCOME_X_CAR: float,
        CUTENURE:         int,
        IS_OWNER:         int,
        MORTGAGE_FLAG:    int,
        HAS_MORTGAGE:     int,
        NUM_CARS:         int,
        annual_income:    Optional[float] = None,
    ) -> dict:
        """
        Predict disposable income for a single household.

        The model internally predicts spend_ratio, but that is an intermediate
        result. The final output of interest is:

            disposable_income = annual_income * (1 - spend_ratio)

        Parameters
        ----------
        LOG_INCOME, INCOME_X_SIZE, LOG_INCOME_X_CAR : float
            Income-derived features. Use build_features() to compute these
            automatically from a plain annual_income figure.
        CUTENURE, IS_OWNER, MORTGAGE_FLAG, HAS_MORTGAGE, NUM_CARS : int
            Demographic / tenure features.
        annual_income : float, optional
            The household's actual annual income in $. If provided, disposable
            income is computed directly. If omitted, it is back-calculated from
            LOG_INCOME via exp(LOG_INCOME) - 1.

        Returns
        -------
        dict with keys:
            spend_ratio       — predicted fraction of income spent (e.g. 0.72)
            annual_income     — income used for the disposable income calculation
            disposable_income — annual_income * (1 - spend_ratio)
        """
        self._require("_active_model")
        feats_in = {
            "LOG_INCOME": LOG_INCOME, "INCOME_X_SIZE": INCOME_X_SIZE,
            "LOG_INCOME_X_CAR": LOG_INCOME_X_CAR, "CUTENURE": CUTENURE,
            "IS_OWNER": IS_OWNER, "MORTGAGE_FLAG": MORTGAGE_FLAG,
            "HAS_MORTGAGE": HAS_MORTGAGE, "NUM_CARS": NUM_CARS,
        }
        row = np.array([[feats_in.get(f, 0.0) for f in self._active_feats]], dtype=float)
        spend_ratio = float(self._active_model.predict(row)[0])

        # Recover income from LOG_INCOME if not explicitly supplied
        income = annual_income if annual_income is not None else float(np.expm1(LOG_INCOME))

        disposable = income * (1.0 - spend_ratio)

        return {
            "spend_ratio":       round(spend_ratio, 4),
            "annual_income":     round(income, 2),
            "disposable_income": round(disposable, 2),
        }

    def predict_df(self, df: pd.DataFrame, income_col: Optional[str] = None) -> pd.DataFrame:
        """
        Predict disposable income for a DataFrame of households.

        The DataFrame must contain columns matching the active model's feature list.
        Returns a DataFrame with three new columns:
            spend_ratio, annual_income, disposable_income

        Parameters
        ----------
        df         : DataFrame containing the 8 feature columns.
        income_col : Optional column name in df holding actual annual income.
                     If omitted, income is back-calculated from LOG_INCOME.
        """
        self._require("_active_model")
        missing = [f for f in self._active_feats if f not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        X = df[self._active_feats].values.astype(float)
        spend_ratios = self._active_model.predict(X)

        if income_col and income_col in df.columns:
            incomes = df[income_col].values.astype(float)
        else:
            incomes = np.expm1(df["LOG_INCOME"].values.astype(float))

        return pd.DataFrame({
            "spend_ratio":       spend_ratios.round(4),
            "annual_income":     incomes.round(2),
            "disposable_income": (incomes * (1.0 - spend_ratios)).round(2),
        }, index=df.index)

    def swap_model(self, name: str) -> "SpendingRatioPredictor":
        """
        Switch which trained model is used by predict() and predict_df().
        Call summary() to see all available model names.
        """
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.trained_models.keys())}")
        self._active_model = self.trained_models[name]
        self._active_name  = name
        # two_stage uses s2_feats; everything else uses final_feats
        self._active_feats = self.s2_feats if name == "two_stage" else self.final_feats
        print(f"  Active model  -> '{name}'")
        return self

      
    # Summary, save, load, run_all
      

    def summary(self) -> "SpendingRatioPredictor":
        """Print a complete results table."""
        print("\n" + "=" * 65)
        print("  EXPERIMENT SUMMARY")
        print("=" * 65)
        if self.final_feats:
            print(f"  Optimal features ({self.best_n}): {self.final_feats}")
        if self.model_results:
            print(f"\n  {'Model':<30} {'CV R²':>8} {'std':>7}")
            print("  " + "-" * 48)
            for name, res in sorted(self.model_results.items(), key=lambda x: -x[1]["R2"]):
                marker = "  ◀ active" if name == self._active_name else ""
                print(f"  {name:<30} {res['R2']:>8.4f} {res['R2_std']:>7.4f}{marker}")
        if self.s1_r2 is not None:
            print(f"\n  Two-stage  Stage1 R²={self.s1_r2:.4f}  Stage2 R²={self.s2_r2:.4f}")
        if self._active_model and hasattr(self._active_model, "feature_importances_"):
            fi = pd.Series(
                self._active_model.feature_importances_, index=self._active_feats
            ).sort_values(ascending=False)
            print(f"\n  Feature importances ({self._active_name}):")
            for feat, imp in fi.items():
                bar = "█" * int(imp * 40)
                print(f"    {feat:<25} {imp:.4f}  {bar}")
        print("=" * 65)
        return self

    def save(self, path: str) -> "SpendingRatioPredictor":
        """Pickle the entire experiment state (all models, results, data) to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✓ Saved  -> {path}")
        return self

    @staticmethod
    def load(path: str) -> "SpendingRatioPredictor":
        """Load a saved experiment from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        r2 = obj.model_results.get(obj._active_name, {}).get("R2", "?")
        print(f"  ✓ Loaded ← {path}  (active: '{obj._active_name}', R²={r2})")
        return obj

    def run_all(self, verbose: bool = True) -> "SpendingRatioPredictor":
        """Convenience: run every step in order."""
        return (
            self
            .load_data(verbose=verbose)
            .run_importance(verbose=verbose)
            .run_feature_scan(verbose=verbose)
            .run_all_models(verbose=verbose)
            .tune_best(verbose=verbose)
            .run_two_stage(verbose=verbose)
            .plot()
            .summary()
        )

      
    # Static helpers
      

    @staticmethod
    def build_features(
        annual_income:  float,
        household_size: int,
        num_cars:       int,
        cutenure:       int,
    ) -> dict:
        """
        Compute all 8 predict() features from plain human inputs.
        Pass the result directly with **build_features(...).
        """
        log_inc = np.log1p(annual_income)
        return {
            "LOG_INCOME":       log_inc,
            "INCOME_X_SIZE":    annual_income * household_size,
            "LOG_INCOME_X_CAR": log_inc * num_cars,
            "CUTENURE":         cutenure,
            "IS_OWNER":         int(cutenure in (1, 2, 3)),
            "MORTGAGE_FLAG":    int(cutenure == 1),
            "HAS_MORTGAGE":     int(cutenure == 1),
            "NUM_CARS":         num_cars,
        }

    @staticmethod
    def encode_tenure(tenure_str: str) -> int:
        """Convert a readable tenure string to its CUTENURE code."""
        mapping = {
            "own_mortgage": 1, "own_clear": 2, "own_unknown": 3,
            "rent_no_cash": 4, "renter": 5,
        }
        key = tenure_str.lower().strip()
        if key not in mapping:
            raise ValueError(f"Unknown tenure '{tenure_str}'. Options: {list(mapping.keys())}")
        return mapping[key]

    def _require(self, attr: str):
        step_map = {
            "hh":              "load_data()",
            "imp_df":          "run_importance()",
            "final_feats":     "run_feature_scan()",
            "best_model_name": "run_all_models()",
            "_active_model":   "run_all_models()",
        }
        if getattr(self, attr, None) is None:
            raise RuntimeError(f"Must call {step_map.get(attr, attr)} first.")                                                              

if __name__ == "__main__":
    # Run full experiment
    p = SpendingRatioPredictor(data_folder="intrvw20/").run_all()
    p.save("spend_ratio_model.pkl")

    # Predict with best tuned model (default)
    feats = SpendingRatioPredictor.build_features(
        annual_income=95_000, household_size=3, num_cars=2,
        cutenure=SpendingRatioPredictor.encode_tenure("own_mortgage"),
    )
    result = p.predict(**feats, annual_income=95_000)
    print(f"\n  Spend ratio      : {result['spend_ratio']:.4f}")
    print(f"  Annual income    : ${result['annual_income']:,.0f}")
    print(f"  Disposable income: ${result['disposable_income']:,.0f}")

    # Compare the same inputs across all trained models
    print("\n  Cross-model comparison:")
    for name in p.trained_models:
        p.swap_model(name)
        r = p.predict(**feats, annual_income=95_000)
        print(f"    {name:<30} spend_ratio={r['spend_ratio']:.4f}  disposable=${r['disposable_income']:,.0f}")
