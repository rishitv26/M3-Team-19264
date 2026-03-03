"""
GamblingLossPredictor — Experimental + Production Class
========================================================
Q2: Know the Spread — M3 Challenge 2026

Mirrors the SpendingRatioPredictor (Q1) class structure exactly.

Pipeline:
  • load_data()        — load q2_simulation_data.csv, engineer all features
  • run_all_models()   — Ridge, Random Forest, Gradient Boosting with 5-fold CV
  • tune_best()        — RandomizedSearchCV on the best model
  • plot()             — all diagnostic charts
  • summary()          — print full results table
  • predict()          — single individual (plain human inputs)
  • predict_df()       — DataFrame of individuals
  • swap_model()       — switch which trained model predict() uses
  • save() / load()    — persist entire experiment state to disk
  • run_all()          — one-call convenience

Feature rounds (best = Round 3, all three layers):
  Round 1 — Demographics:    age, gender, income
  Round 2 — + Disposable DI: from Q1 model output
  Round 3 — + Behavioral:    risk tolerance, wager intensity (log F_i)

Usage
-----
    from q2_loss_predictor import GamblingLossPredictor

    # ── Run full experiment ──────────────────────────────────────────
    g = GamblingLossPredictor()
    g.load_data()
    g.run_all_models()
    g.tune_best()
    g.plot()
    g.summary()
    g.save("gambling_loss_model.pkl")

    # ── Or run everything in one call ────────────────────────────────
    g = GamblingLossPredictor().run_all()
    g.save("gambling_loss_model.pkl")

    # ── Load and predict ─────────────────────────────────────────────
    g = GamblingLossPredictor.load("gambling_loss_model.pkl")

    result = g.predict(
        age=28, male=1, income=65_000,
        disposable_income=20_000, risk_tolerance="high",
    )
    print(result)
    # {'predicted_loss': 412, 'pi_low': 80, 'pi_high': 1240,
    #  'loss_pct_di': 2.1, 'moderate_harm': False, 'severe_harm': False}

    # ── Use build_features() for DataFrame predictions ───────────────
    feats = GamblingLossPredictor.build_features(
        age=45, male=1, income=150_000,
        disposable_income=55_000, risk_tolerance="extreme",
    )
    result = g.predict(**feats)

    # ── Swap to a different trained model ────────────────────────────
    g.swap_model("Random Forest")
    result = g.predict(age=28, male=1, income=65_000,
                       disposable_income=20_000, risk_tolerance="medium")

    # ── Predict for a DataFrame ──────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame([
        {"age":22,"male":1,"income":32_000,"disposable_income":9_000, "risk_tolerance":"high"},
        {"age":55,"male":0,"income":80_000,"disposable_income":25_000,"risk_tolerance":"low"},
    ])
    preds = g.predict_df(df)
    print(preds)
"""

from __future__ import annotations

import pickle
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

# Round 3 feature set — demographics + disposable income + behavioral
ALL_FEATURES = [
    # Demographics (Round 1)
    "age", "age_sq", "male", "log_income",
    "age_1834", "age_3549", "age_5064",
    "young_male",
    # Disposable income (Round 2)
    "log_di",
    # Behavioral (Round 3)
    "log_wager_frac", "risk_medium", "risk_high", "risk_extreme",
]

DEMO_FEATURES = [
    "age", "age_sq", "male", "log_income",
    "age_1834", "age_3549", "age_5064", "young_male",
]

DEMO_DI_FEATURES = DEMO_FEATURES + ["log_di"]

FEATURE_LABELS = {
    "log_wager_frac": "Wager Intensity (log F)",
    "risk_extreme":   "Risk: Extreme",
    "risk_high":      "Risk: High",
    "risk_medium":    "Risk: Medium",
    "log_di":         "Disposable Income (log)",
    "log_income":     "Gross Income (log)",
    "young_male":     "Young × Male",
    "male":           "Gender (Male)",
    "age_1834":       "Age 18–34",
    "age_3549":       "Age 35–49",
    "age_5064":       "Age 50–64",
    "age":            "Age",
    "age_sq":         "Age²",
}

PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


# ── Class ──────────────────────────────────────────────────────────────────────

class GamblingLossPredictor:
    """
    Full experimental pipeline + prediction interface for annual gambling loss.

    Trained on structural Monte Carlo simulation output (q2_simulation_data.csv),
    calibrated to AGA hold rates, Siena participation data, and Pareto
    concentration literature. Prediction target: net annual gambling loss ($)
    conditional on participation.
    """

    def __init__(
        self,
        sim_file:     str = "q2_simulation_data.csv",
        cv_folds:     int = 5,
        random_state: int = 42,
    ):
        self.sim_file     = sim_file
        self.cv_folds     = cv_folds
        self.random_state = random_state

        # Data
        self.gamblers:      Optional[pd.DataFrame] = None   # load_data()
        self.risk_quartiles: Optional[dict]        = None   # load_data()
        self.residuals:      Optional[np.ndarray]  = None   # run_all_models()

        # Model results
        self.model_results:  dict = {}   # name → {R2, R2_std, MAE}
        self.trained_models: dict = {}   # name → fitted model object
        self.best_model_name: Optional[str] = None

        # Tuning
        self.tuned_model:  Optional[object] = None
        self.tuned_r2:     Optional[float]  = None
        self.tuned_params: Optional[dict]   = None

        # Active model used by predict()
        self._active_model: Optional[object] = None
        self._active_feats: Optional[list]   = None
        self._active_name:  Optional[str]    = None

    # =========================================================================
    # STEP 1 — Load & engineer
    # =========================================================================

    def load_data(self, verbose: bool = True) -> "GamblingLossPredictor":
        """
        Load simulation output CSV, restrict to gamblers, engineer all features.

        Feature engineering follows the iterative structure:
          Round 1 (demographics):  age²,  log(income), age group dummies,
                                   gender×age interaction
          Round 2 (+ DI):          log(disposable_income)
          Round 3 (+ behavioral):  log(wager_frac),  risk quartile dummies
        """
        try:
            df = pd.read_csv(self.sim_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"'{self.sim_file}' not found. "
                "Run q2_gambling_model.py first to generate simulation data."
            )

        if verbose:
            print(f"Loaded simulation data: {len(df):,} agents")

        # Restrict to gamblers — Stage 2 of two-part model (Duan et al. 1983)
        gamblers = df[df["gambles"] == 1].copy()

        if verbose:
            print(f"Gamblers (model population): {len(gamblers):,} "
                  f"({len(gamblers)/len(df)*100:.1f}%)")

        # ── Round 1: demographics
        gamblers["age_sq"]    = gamblers["age"] ** 2
        gamblers["log_income"]= np.log(gamblers["income"].clip(lower=1))
        gamblers["age_1834"]  = ((gamblers["age"] >= 18) & (gamblers["age"] <= 34)).astype(int)
        gamblers["age_3549"]  = ((gamblers["age"] >= 35) & (gamblers["age"] <= 49)).astype(int)
        gamblers["age_5064"]  = ((gamblers["age"] >= 50) & (gamblers["age"] <= 64)).astype(int)
        gamblers["young_male"]= ((gamblers["age"] <= 34) & (gamblers["male"] == 1)).astype(int)

        # ── Round 2: disposable income
        gamblers["log_di"] = np.log(gamblers["disposable_income"].clip(lower=1))

        # ── Round 3: behavioral — risk tolerance from wager_frac quartiles
        q25 = gamblers["wager_frac"].quantile(0.25)
        q50 = gamblers["wager_frac"].quantile(0.50)
        q75 = gamblers["wager_frac"].quantile(0.75)

        self.risk_quartiles = {"q25": q25, "q50": q50, "q75": q75}

        gamblers["risk_medium"]  = ((gamblers["wager_frac"] > q25) &
                                    (gamblers["wager_frac"] <= q50)).astype(int)
        gamblers["risk_high"]    = ((gamblers["wager_frac"] > q50) &
                                    (gamblers["wager_frac"] <= q75)).astype(int)
        gamblers["risk_extreme"] = (gamblers["wager_frac"] > q75).astype(int)
        gamblers["log_wager_frac"] = np.log(gamblers["wager_frac"].clip(lower=1e-6))

        self.gamblers = gamblers

        if verbose:
            print(f"\n── Gambler Population Snapshot ─────────────────────────────────────")
            print(f"  Mean age:               {gamblers['age'].mean():.1f}")
            print(f"  Male:                   {gamblers['male'].mean()*100:.1f}%")
            print(f"  Mean gross income:      ${gamblers['income'].mean():,.0f}")
            print(f"  Mean disposable income: ${gamblers['disposable_income'].mean():,.0f}")
            print(f"  Mean net loss:          ${gamblers['net_loss'].mean():,.0f}")
            print(f"  Median net loss:        ${gamblers['net_loss'].median():,.0f}")
            print(f"  Risk quartile F bounds: "
                  f"low≤{q25*100:.2f}%  med≤{q50*100:.2f}%  high≤{q75*100:.2f}%  extreme>{q75*100:.2f}%")

        return self

    # =========================================================================
    # STEP 2 — Train all models (3 feature rounds × 3 models)
    # =========================================================================

    def run_all_models(self, verbose: bool = True) -> "GamblingLossPredictor":
        """
        Train and cross-validate Ridge, Random Forest, and Gradient Boosting
        on all three feature rounds. Best model on Round 3 becomes active.
        """
        self._require("gamblers")
        gamblers = self.gamblers
        y = gamblers["net_loss"]
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        rounds = {
            "Round1_Ridge":    (DEMO_FEATURES,    Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))])),
            "Round2_Ridge":    (DEMO_DI_FEATURES, Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))])),
            "Round3_Ridge":    (ALL_FEATURES,     Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))])),
            "Round3_RF":       (ALL_FEATURES,     RandomForestRegressor(
                                    n_estimators=300, max_depth=8,
                                    random_state=self.random_state, n_jobs=-1)),
            "Round3_GBM":      (ALL_FEATURES,     GradientBoostingRegressor(
                                    n_estimators=300, max_depth=4, learning_rate=0.05,
                                    random_state=self.random_state)),
        }

        # Human-readable display names
        DISPLAY = {
            "Round1_Ridge": "Ridge — Demographics only",
            "Round2_Ridge": "Ridge — Demo + DI",
            "Round3_Ridge": "Ridge — Demo + DI + Behavioral",
            "Round3_RF":    "Random Forest — All features",
            "Round3_GBM":   "Gradient Boosting — All features",
        }

        if verbose:
            print(f"\n── Model Comparison (5-fold CV) ─────────────────────────────────────")
            print(f"\n  {'Model':<38} {'R²':>8} {'std':>7} {'MAE':>12}")
            print("  " + "-" * 70)

        for key, (feats, model) in rounds.items():
            X = gamblers[feats].fillna(0)
            r2  = cross_val_score(model, X, y, cv=kf, scoring="r2")
            mae = -cross_val_score(model, X, y, cv=kf,
                                   scoring="neg_mean_absolute_error")
            self.model_results[DISPLAY[key]] = {
                "R2": r2.mean(), "R2_std": r2.std(), "MAE": mae.mean(),
                "feats": feats,
            }
            model.fit(X, y)
            self.trained_models[DISPLAY[key]] = (model, feats)
            if verbose:
                print(f"  {DISPLAY[key]:<38} {r2.mean():>8.4f} "
                      f"{r2.std():>7.4f} ${mae.mean():>10,.0f}")

        # Best model = highest R² among Round 3 models
        r3_models = {k: v for k, v in self.model_results.items() if "All features" in k or "Behavioral" in k}
        self.best_model_name = max(r3_models, key=lambda k: r3_models[k]["R2"])

        best_model, best_feats = self.trained_models[self.best_model_name]
        self._active_model = best_model
        self._active_feats = best_feats
        self._active_name  = self.best_model_name

        # Store held-out residuals for prediction intervals
        X_full = gamblers[best_feats].fillna(0)
        _, X_test, _, y_test = train_test_split(
            X_full, y, test_size=0.2, random_state=self.random_state)
        y_pred = best_model.predict(X_test)
        self.residuals = (y_test.values - y_pred)

        if verbose:
            print(f"\n  Best model: {self.best_model_name}  "
                  f"R²={self.model_results[self.best_model_name]['R2']:.4f}")

        return self

    # =========================================================================
    # STEP 3 — Hyperparameter tuning
    # =========================================================================

    def tune_best(self, verbose: bool = True) -> "GamblingLossPredictor":
        """Tune the best model from run_all_models() via RandomizedSearchCV."""
        self._require("best_model_name")
        gamblers = self.gamblers
        y = gamblers["net_loss"]
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        X = gamblers[ALL_FEATURES].fillna(0)

        name = self.best_model_name
        if verbose:
            print(f"\n── Tuning: {name} ───────────────────────────────────────────────────")

        if "Random Forest" in name:
            param_dist = {
                "n_estimators":     [200, 300, 500],
                "max_depth":        [6, 8, 12, None],
                "min_samples_leaf": [1, 2, 5],
                "max_features":     ["sqrt", "log2", 0.5],
            }
            base = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        elif "Boosting" in name or "GBM" in name:
            param_dist = {
                "n_estimators":  [200, 300, 500],
                "max_depth":     [3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample":     [0.7, 0.8, 1.0],
            }
            base = GradientBoostingRegressor(random_state=self.random_state)
        else:
            if verbose:
                print(f"  No tuning grid for '{name}' — skipping.")
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
        self.model_results[tuned_name]  = {
            "R2": self.tuned_r2, "R2_std": 0,
            "MAE": 0, "feats": ALL_FEATURES,
        }
        self.trained_models[tuned_name] = (self.tuned_model, ALL_FEATURES)

        # Update active model + refresh residuals
        self._active_model = self.tuned_model
        self._active_feats = ALL_FEATURES
        self._active_name  = tuned_name

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state)
        self.residuals = y_test.values - self.tuned_model.predict(X_test)

        if verbose:
            print(f"  Tuned R²: {self.tuned_r2:.4f}")
            print(f"  Best params: {self.tuned_params}")

        return self

    # =========================================================================
    # STEP 4 — Plots
    # =========================================================================

    def plot(self, save: bool = True) -> "GamblingLossPredictor":
        """Generate and save all diagnostic visualisation panels."""
        self._require("gamblers")
        gamblers = self.gamblers
        fmt_d = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Q2: GamblingLossPredictor — Diagnostic Plots",
                     fontsize=15, fontweight="bold", y=1.01)

        # 1: Loss distribution
        ax = axes[0, 0]
        pos = gamblers["net_loss"][gamblers["net_loss"] > 0]
        ax.hist(pos, bins=np.logspace(0, 5, 60),
                color=PALETTE[0], edgecolor="white", alpha=0.85)
        ax.set_xscale("log")
        ax.axvline(pos.median(), color="orange", linestyle="-.",
                   linewidth=1.5, label=f"Median: ${pos.median():,.0f}")
        ax.axvline(pos.quantile(0.95), color="red", linestyle="--",
                   linewidth=1.5, label=f"P95: ${pos.quantile(0.95):,.0f}")
        ax.set_title("Net Loss Distribution (Log Scale)")
        ax.set_xlabel("Annual Net Loss ($, log scale)")
        ax.set_ylabel("Count"); ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(fmt_d)

        # 2: Mean loss by age group
        ax = axes[0, 1]
        age_loss = gamblers.groupby("age_group", observed=True)["net_loss"].mean()
        age_loss.plot(kind="bar", ax=ax, color=PALETTE[1], edgecolor="white", width=0.7)
        ax.set_title("Mean Annual Loss by Age Group")
        ax.set_xlabel(""); ax.set_ylabel("Mean Net Loss ($)")
        ax.yaxis.set_major_formatter(fmt_d)
        ax.tick_params(axis="x", rotation=30)

        # 3: Model R² comparison
        ax = axes[0, 2]
        disp   = {k: v for k, v in self.model_results.items() if v["R2"] > 0}
        names  = list(disp.keys())
        r2vals = [disp[n]["R2"] for n in names]
        colors = [PALETTE[2] if n == self._active_name else PALETTE[0] for n in names]
        bars   = ax.bar(range(len(names)), r2vals, color=colors, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" — ", "\n") for n in names], fontsize=7)
        ax.set_title("CV R² by Model (green = active)")
        ax.set_ylabel("R²"); ax.set_ylim(0, max(r2vals) * 1.2)
        for bar, val in zip(bars, r2vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8)
        ax.tick_params(axis="x", rotation=15)

        # 4: Feature importance (permutation)
        ax = axes[1, 0]
        if self._active_model is not None and self.residuals is not None:
            gamblers_clean = gamblers[ALL_FEATURES].fillna(0)
            _, X_test, _, y_test = train_test_split(
                gamblers_clean, gamblers["net_loss"],
                test_size=0.2, random_state=self.random_state)
            perm = permutation_importance(
                self._active_model, X_test, y_test,
                n_repeats=15, random_state=self.random_state, n_jobs=-1)
            imp = pd.Series(perm.importances_mean, index=ALL_FEATURES).sort_values()
            labels = [FEATURE_LABELS.get(f, f) for f in imp.index]
            colors_imp = [PALETTE[3] if v >= imp.quantile(0.75) else PALETTE[0]
                          for v in imp.values]
            ax.barh(labels, imp.values, color=colors_imp, edgecolor="white")
            ax.set_title(f"Permutation Importance\n({self._active_name})")
            ax.set_xlabel("Mean ΔR² when permuted")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

        # 5: Predicted vs actual
        ax = axes[1, 1]
        if self._active_model is not None:
            X_full = gamblers[self._active_feats].fillna(0)
            _, X_test, _, y_test = train_test_split(
                X_full, gamblers["net_loss"],
                test_size=0.2, random_state=self.random_state)
            y_pred = self._active_model.predict(X_test)
            lim = float(np.percentile(np.abs(y_test), 95))
            ax.scatter(y_test, y_pred, alpha=0.1, s=5, color=PALETTE[0])
            ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("Actual Net Loss ($)")
            ax.set_ylabel("Predicted Net Loss ($)")
            r2_test = r2_score(y_test, y_pred)
            mae_test = mean_absolute_error(y_test, y_pred)
            ax.set_title(f"Predicted vs Actual\nR²={r2_test:.3f}  MAE=${mae_test:,.0f}")
            ax.xaxis.set_major_formatter(fmt_d)
            ax.yaxis.set_major_formatter(fmt_d)
            ax.grid(alpha=0.2)

        # 6: Loss as % of DI by income tier
        ax = axes[1, 2]
        pct_income = gamblers.groupby("income_tier", observed=True)["loss_pct_di"].mean()
        pct_income.plot(kind="bar", ax=ax, color=PALETTE[4], edgecolor="white", width=0.7)
        ax.set_title("Mean Loss as % of DI by Income Tier\n(lower income = proportionally more harmed)")
        ax.set_xlabel(""); ax.set_ylabel("Mean Loss / DI (%)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        if save:
            plt.savefig("q2_predictor_diagnostics.png", dpi=150, bbox_inches="tight")
            print("  ✓ Saved: q2_predictor_diagnostics.png")
        plt.show()
        return self

    # =========================================================================
    # Prediction interface
    # =========================================================================

    def predict(
        self,
        age:                float,
        male:               int,
        income:             float,
        disposable_income:  float,
        risk_tolerance:     str = "medium",
    ) -> dict:
        """
        Predict expected annual net gambling loss for a single individual.

        Parameters
        ----------
        age               : Age in years (18+)
        male              : 1 = male, 0 = female
        income            : Annual gross income ($)
        disposable_income : Annual disposable income ($) — from Q1 model output
        risk_tolerance    : Betting intensity tier:
                            "low"     — bottom quartile of wager fraction (F ≤ Q25)
                            "medium"  — Q25 < F ≤ Q50
                            "high"    — Q50 < F ≤ Q75
                            "extreme" — F > Q75 (top 25% most intensive bettors)

        Returns
        -------
        dict with keys:
            predicted_loss   — point estimate of annual net loss ($)
            pi_low           — 10th percentile of 80% prediction interval ($)
            pi_high          — 90th percentile of 80% prediction interval ($)
            loss_pct_di      — predicted loss as % of disposable income
            moderate_harm    — True if predicted loss > 5% of DI
            severe_harm      — True if predicted loss > 20% of DI
            active_model     — name of the model used for this prediction
        """
        self._require("_active_model")
        feats = self.build_features(
            age=age, male=male, income=income,
            disposable_income=disposable_income,
            risk_tolerance=risk_tolerance,
            _quartiles=self.risk_quartiles,
        )
        row = np.array([[feats[f] for f in self._active_feats]], dtype=float)
        point = float(self._active_model.predict(row)[0])

        # 80% prediction interval from held-out residuals
        p10 = point + float(np.percentile(self.residuals, 10))
        p90 = point + float(np.percentile(self.residuals, 90))

        return {
            "predicted_loss": round(max(point, 0), 2),
            "pi_low":         round(max(p10, 0), 2),
            "pi_high":        round(max(p90, 0), 2),
            "loss_pct_di":    round(point / max(disposable_income, 1) * 100, 2),
            "moderate_harm":  point > 0.05 * disposable_income,
            "severe_harm":    point > 0.20 * disposable_income,
            "active_model":   self._active_name,
        }

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict annual net gambling loss for a DataFrame of individuals.

        The DataFrame must contain columns:
            age, male, income, disposable_income, risk_tolerance

        Returns the input DataFrame with five new columns appended:
            predicted_loss, pi_low, pi_high, loss_pct_di,
            moderate_harm, severe_harm
        """
        self._require("_active_model")
        required = ["age", "male", "income", "disposable_income", "risk_tolerance"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        results = []
        for _, row in df.iterrows():
            r = self.predict(
                age=row["age"], male=row["male"],
                income=row["income"],
                disposable_income=row["disposable_income"],
                risk_tolerance=str(row["risk_tolerance"]),
            )
            results.append(r)

        out = df.copy()
        for key in ["predicted_loss","pi_low","pi_high",
                    "loss_pct_di","moderate_harm","severe_harm"]:
            out[key] = [r[key] for r in results]
        return out

    def swap_model(self, name: str) -> "GamblingLossPredictor":
        """
        Switch which trained model is used by predict() and predict_df().
        Call summary() to see all available model names.
        """
        if name not in self.trained_models:
            raise ValueError(
                f"Model '{name}' not found. "
                f"Available: {list(self.trained_models.keys())}"
            )
        model, feats = self.trained_models[name]
        self._active_model = model
        self._active_feats = feats
        self._active_name  = name

        # Refresh residuals for new model
        gamblers = self.gamblers
        X_full = gamblers[feats].fillna(0)
        _, X_test, _, y_test = train_test_split(
            X_full, gamblers["net_loss"],
            test_size=0.2, random_state=self.random_state)
        self.residuals = y_test.values - model.predict(X_test)

        print(f"  Active model → '{name}'")
        return self

    # =========================================================================
    # Summary, save, load, run_all
    # =========================================================================

    def summary(self) -> "GamblingLossPredictor":
        """Print a complete results table."""
        print("\n" + "=" * 70)
        print("  EXPERIMENT SUMMARY — GamblingLossPredictor")
        print("=" * 70)
        if self.model_results:
            print(f"\n  {'Model':<42} {'CV R²':>8} {'std':>7} {'MAE':>12}")
            print("  " + "-" * 72)
            for name, res in sorted(self.model_results.items(),
                                    key=lambda x: -x[1]["R2"]):
                marker = "  ◀ active" if name == self._active_name else ""
                mae_str = f"${res['MAE']:,.0f}" if res["MAE"] > 0 else "  —"
                print(f"  {name:<42} {res['R2']:>8.4f} "
                      f"{res['R2_std']:>7.4f} {mae_str:>12}{marker}")

        if self.tuned_params:
            print(f"\n  Tuned params: {self.tuned_params}")

        if self._active_model and hasattr(self._active_model, "feature_importances_"):
            fi = pd.Series(
                self._active_model.feature_importances_,
                index=self._active_feats,
            ).sort_values(ascending=False)
            print(f"\n  Feature importances ({self._active_name}):")
            for feat, imp in fi.items():
                label = FEATURE_LABELS.get(feat, feat)
                bar   = "█" * int(imp * 60)
                print(f"    {label:<30} {imp:.4f}  {bar}")

        if self.risk_quartiles:
            q = self.risk_quartiles
            print(f"\n  Risk tolerance tiers (wager fraction of DI):")
            print(f"    low     : F ≤ {q['q25']*100:.2f}%")
            print(f"    medium  : {q['q25']*100:.2f}% < F ≤ {q['q50']*100:.2f}%")
            print(f"    high    : {q['q50']*100:.2f}% < F ≤ {q['q75']*100:.2f}%")
            print(f"    extreme : F > {q['q75']*100:.2f}%")

        print("=" * 70)
        return self

    def save(self, path: str) -> "GamblingLossPredictor":
        """Pickle the entire experiment state to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✓ Saved → {path}")
        return self

    @staticmethod
    def load(path: str) -> "GamblingLossPredictor":
        """Load a saved experiment from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        r2 = obj.model_results.get(obj._active_name, {}).get("R2", "?")
        print(f"  ✓ Loaded ← {path}  (active: '{obj._active_name}', R²={r2})")
        return obj

    def run_all(self, verbose: bool = True) -> "GamblingLossPredictor":
        """Convenience: run every step in order."""
        return (
            self
            .load_data(verbose=verbose)
            .run_all_models(verbose=verbose)
            .tune_best(verbose=verbose)
            .plot()
            .summary()
            .save("gambling_loss_model.pkl")
        )

    # =========================================================================
    # Static helpers
    # =========================================================================

    @staticmethod
    def build_features(
        age:               float,
        male:              int,
        income:            float,
        disposable_income: float,
        risk_tolerance:    str   = "medium",
        _quartiles:        Optional[dict] = None,
    ) -> dict:
        """
        Compute all feature-engineered inputs from plain human inputs.
        Pass the result directly with **build_features(...) to predict().

        Parameters
        ----------
        age               : Age in years (18+)
        male              : 1 = male, 0 = female
        income            : Annual gross income ($)
        disposable_income : Annual disposable income ($)
        risk_tolerance    : "low" / "medium" / "high" / "extreme"
        _quartiles        : Internal — passed automatically when called from
                            predict(). If calling standalone, omit this.
        """
        # Map risk tolerance to a representative log_wager_frac
        # using the population quartile bounds calibrated during load_data().
        # If quartiles not available (standalone use), use typical values.
        if _quartiles:
            q25, q50, q75 = _quartiles["q25"], _quartiles["q50"], _quartiles["q75"]
        else:
            # Fallback: typical values from simulation calibration
            q25, q50, q75 = 0.0007, 0.0037, 0.0180

        risk_map = {
            "low":     np.log(q25 * 0.5),
            "medium":  np.log((q25 + q50) / 2),
            "high":    np.log((q50 + q75) / 2),
            "extreme": np.log(q75 * 2.5),
        }
        risk_key = risk_tolerance.lower().strip()
        if risk_key not in risk_map:
            raise ValueError(
                f"Unknown risk_tolerance '{risk_tolerance}'. "
                f"Options: {list(risk_map.keys())}"
            )
        lwf = risk_map[risk_key]

        return {
            "age":            float(age),
            "age_sq":         float(age) ** 2,
            "male":           int(male),
            "log_income":     np.log(max(income, 1)),
            "age_1834":       int(18 <= age <= 34),
            "age_3549":       int(35 <= age <= 49),
            "age_5064":       int(50 <= age <= 64),
            "young_male":     int(age <= 34 and male == 1),
            "log_di":         np.log(max(disposable_income, 1)),
            "log_wager_frac": lwf,
            "risk_medium":    int(risk_key == "medium"),
            "risk_high":      int(risk_key == "high"),
            "risk_extreme":   int(risk_key == "extreme"),
        }

    @staticmethod
    def encode_risk(risk_str: str) -> str:
        """
        Validate and normalise a risk tolerance string.
        Accepts flexible aliases (e.g. 'hi', 'HIGH', 'moderate').
        """
        mapping = {
            "low": "low", "l": "low", "conservative": "low",
            "medium": "medium", "med": "medium", "m": "medium", "moderate": "medium",
            "high": "high", "hi": "high", "h": "high", "aggressive": "high",
            "extreme": "extreme", "x": "extreme", "max": "extreme",
            "very high": "extreme", "problem": "extreme",
        }
        key = risk_str.lower().strip()
        if key not in mapping:
            raise ValueError(
                f"Unknown risk string '{risk_str}'. "
                f"Options: {list(set(mapping.values()))}"
            )
        return mapping[key]

    def _require(self, attr: str):
        step_map = {
            "gamblers":       "load_data()",
            "risk_quartiles": "load_data()",
            "best_model_name":"run_all_models()",
            "_active_model":  "run_all_models()",
            "residuals":      "run_all_models()",
        }
        if getattr(self, attr, None) is None:
            raise RuntimeError(
                f"Must call {step_map.get(attr, attr)} first."
            )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Run full experiment
    g = GamblingLossPredictor().run_all()
    g.save("gambling_loss_model.pkl")

    # ── Single predictions across demographic profiles
    print("\n" + "="*70)
    print("INDIVIDUAL PREDICTIONS — cross-model comparison")
    print("="*70)

    profiles = [
        {"age":22, "male":1, "income":32_000,  "di":9_000,  "risk":"high",    "label":"Young male, low income, high risk"},
        {"age":28, "male":1, "income":65_000,  "di":20_000, "risk":"medium",  "label":"Young male, median income, medium risk"},
        {"age":38, "male":0, "income":72_000,  "di":23_000, "risk":"low",     "label":"Mid-age female, mid income, low risk"},
        {"age":45, "male":1, "income":150_000, "di":55_000, "risk":"extreme", "label":"Mid-age male, high income, extreme risk"},
        {"age":58, "male":1, "income":80_000,  "di":25_000, "risk":"medium",  "label":"Older male, mid income, medium risk"},
        {"age":68, "male":0, "income":35_000,  "di":10_000, "risk":"low",     "label":"Senior female, low income, low risk"},
    ]

    rows = []
    for p in profiles:
        res = g.predict(p["age"], p["male"], p["income"], p["di"], p["risk"])
        rows.append({
            "Profile":         p["label"],
            "DI":              f"${p['di']:,}",
            "Risk":            p["risk"].capitalize(),
            "Loss (est)":      f"${res['predicted_loss']:,.0f}",
            "80% PI":          f"[${res['pi_low']:,.0f}–${res['pi_high']:,.0f}]",
            "% of DI":         f"{res['loss_pct_di']:.1f}%",
            "Mod. Harm":       "⚠️" if res["moderate_harm"] else "—",
            "Severe Harm":     "🚨" if res["severe_harm"] else "—",
        })

    print(pd.DataFrame(rows).set_index("Profile").to_string())

    # ── Cross-model comparison for one profile
    print("\n── Cross-model comparison: Young male, $65k income, high risk ─────────")
    for model_name in g.trained_models:
        g.swap_model(model_name)
        r = g.predict(28, 1, 65_000, 20_000, "high")
        print(f"  {model_name:<45}  loss=${r['predicted_loss']:,.0f}  "
              f"({r['loss_pct_di']:.1f}% of DI)")

    # ── predict_df example
    print("\n── predict_df example ───────────────────────────────────────────────────")
    g.swap_model(g.best_model_name + " (tuned)")
    sample = pd.DataFrame([
        {"age":22,"male":1,"income":32_000, "disposable_income":9_000,  "risk_tolerance":"high"},
        {"age":38,"male":0,"income":72_000, "disposable_income":23_000, "risk_tolerance":"low"},
        {"age":45,"male":1,"income":150_000,"disposable_income":55_000, "risk_tolerance":"extreme"},
    ])
    out = g.predict_df(sample)
    print(out[["age","male","income","risk_tolerance",
               "predicted_loss","loss_pct_di","moderate_harm"]].to_string())