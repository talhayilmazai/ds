# Plan: DAYTON Energy Consumption — Time-Series Forecasting Notebook (Revised)

## Context

The student is working on the **Energy Consumption Forecasting** capstone project (DSAI 302).
The deliverable is a single Jupyter notebook whose section structure mirrors the IEEE-format
report so the same artifact drives both the *Methods*/*Results* sections of the paper and the
demo for the oral exam.

The notebook must:
- Load `DAYTON_hourly.csv` and run interactive EDA.
- Engineer pure temporal features **and** lag/rolling-mean features.
- Split chronologically (70 / 15 / 15) — no random sampling.
- Tune hyperparameters for **Random Forest, XGBoost, LightGBM** using
  `TimeSeriesSplit` + `RandomizedSearchCV` (no standard K-Fold — that leaks future into past).
- Evaluate on the **held-out test set** with MAE, MAPE, RMSE, MSE, R².
- Produce actual-vs-predicted and feature-importance plots at IEEE figure quality (300 DPI).

### Dataset facts (verified)
- File: `DAYTON_hourly.csv` (121,275 hourly rows, ~3.2 MB)
- Columns: `Datetime` (actual format `YYYY-MM-DD HH:MM:SS`, despite project description),
  `DAYTON_MW`. Range: 2004-12-31 → 2018-01-02.
- The first row of the file is **out of chronological order** — the notebook must
  `sort_values('Datetime')` immediately after load. This is a real foot-gun: skipping the
  sort silently corrupts the train/val/test boundaries.

## Approach

One notebook, `energy_forecasting.ipynb`, at the project root. Markdown cells delimit
sections that map 1:1 to IEEE report sections. Code cells are kept short and focused so the
student can re-run pieces independently. `RANDOM_STATE = 42` everywhere.

### Notebook structure

#### 1. Introduction *(markdown)*
- Problem statement, motivation, dataset description, objectives.
- Stub paragraph the student will expand for the report.

#### 2. Data Analysis and Data Preprocess

**2.1 Setup** *(code)*
Imports: `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.model_selection`
(`TimeSeriesSplit`, `RandomizedSearchCV`), `sklearn.ensemble.RandomForestRegressor`,
`sklearn.metrics`, `xgboost as xgb`, `lightgbm as lgb`. Set seaborn style and a print-friendly
palette per CLAUDE.md.

**2.2 Load & Inspect** *(code)*
- Read CSV, parse `Datetime`, **sort ascending**, drop duplicate timestamps, reset index.
- `df.info()`, `df.describe()`, `df.isna().sum()`, duplicate count, full datetime range,
  expected-vs-actual hour count (gaps check).

**2.3 Exploratory Data Analysis** *(multiple code cells)*
- Full time-series line plot (downsampled daily mean for readability).
- Distribution of `DAYTON_MW` (histogram + KDE).
- Seasonality plots: hourly profile (boxplot of MW by hour-of-day), weekly profile (by
  day-of-week), monthly profile (by month). These three plots both inform feature design and
  give the report's *Data Analysis* section its visual content.
- Brief markdown commentary cells under each plot summarizing observations (peak hours,
  weekday/weekend gap, seasonal swing).

**2.4 Feature Engineering** *(code)*
- Temporal: `hour`, `dayofweek`, `day`, `month`, `quarter`, `dayofyear`, `weekofyear`,
  `year`, `is_weekend`.
- Cyclical encodings for `hour` and `dayofweek` (sin/cos pairs).
- Lag features: `lag_24` (same hour previous day), `lag_168` (same hour previous week).
- Rolling means: `roll_mean_24`, `roll_mean_168` (computed with `.shift(1)` first to
  prevent target leakage from the current hour).
- Drop the leading rows with NaN from the lag/rolling construction (~168 rows).
- Markdown cell explaining lag-feature rationale and the `.shift(1)` to avoid leakage.

**2.5 Chronological Split** *(code)*
- Index-slice the sorted dataframe: 70 % train, 15 % val, 15 % test.
- Print the date range of each subset to make the temporal ordering visible in the report.
- Markdown cell stating *why* random splits are forbidden here (future-information leakage,
  inflated metrics, model that wouldn't generalize to actual deployment).

#### 3. Methods

**3.1 Why `TimeSeriesSplit`?** *(markdown)*
Short rationale: standard K-Fold randomly assigns rows across folds, so a fold's training
data can contain timestamps later than its validation data — that's leakage and produces
optimistic metrics. `TimeSeriesSplit` produces expanding-window folds where each
training window precedes its validation window. This is the only valid CV for time series.

**3.2 CV strategy & search spaces** *(code)*
- `tscv = TimeSeriesSplit(n_splits=5)` — applied **only to the training set** during tuning.
  The held-out validation set still serves two roles: (a) early-stopping signal for XGB/LGBM
  in the final fit, (b) an additional sanity check between training and the test set.
- A `neg_root_mean_squared_error` scoring metric for the search.
- Three search spaces (parameter grids) — kept moderate in size so a `RandomizedSearchCV`
  with `n_iter=20` finishes in reasonable time on a laptop:
  - **RandomForestRegressor**: `n_estimators` ∈ {200, 400, 600},
    `max_depth` ∈ {10, 20, 30, None}, `min_samples_split` ∈ {2, 5, 10},
    `max_features` ∈ {"sqrt", 0.5, 1.0}.
  - **XGBRegressor** (`tree_method="hist"`): `n_estimators` ∈ {300, 500, 800},
    `learning_rate` ∈ {0.03, 0.05, 0.1}, `max_depth` ∈ {4, 6, 8, 10},
    `subsample` ∈ {0.7, 0.85, 1.0}, `colsample_bytree` ∈ {0.7, 0.85, 1.0},
    `min_child_weight` ∈ {1, 3, 5}.
  - **LGBMRegressor**: `n_estimators` ∈ {300, 500, 800},
    `learning_rate` ∈ {0.03, 0.05, 0.1}, `num_leaves` ∈ {31, 63, 127},
    `max_depth` ∈ {-1, 8, 12}, `subsample` ∈ {0.7, 0.85, 1.0},
    `colsample_bytree` ∈ {0.7, 0.85, 1.0}, `min_child_samples` ∈ {10, 20, 40}.

**3.3 Tuning loop** *(code)*
A small helper `tune_model(estimator, param_dist, X_tr, y_tr, n_iter=20)` that returns the
fitted `RandomizedSearchCV` object. Run it three times — one per model. Print best params
and best CV RMSE for each.

**3.4 Final fit** *(code)*
- Take the best estimator from each search.
- For **XGBoost** and **LightGBM**: refit on training data with `eval_set=[(X_val, y_val)]`
  and `early_stopping_rounds=50`. This uses the val set as the early-stopping signal,
  consistent with the original brief and unambiguously avoiding leakage (val is strictly
  after train). For RF (no early stopping), simply use the search's `best_estimator_`
  already refit on the full training set.
- *Why not refit on `train + val`?* Keeping val out of the final fit lets us compare the
  three models on identical val-set RMSE before touching the test set, which is the cleaner
  story for the report.

#### 4. Results

**4.1 Test-set evaluation** *(code)*
A function `evaluate(name, model, X_test, y_test)` returning a dict of MAE, MAPE (in %),
RMSE, MSE, R². Aggregate into a `pd.DataFrame` and print it sorted by RMSE ascending. This
is the headline table for the report.

**4.2 Actual vs Predicted** *(code)*
Plot the **last 14 days** of the test set, actual vs predicted, for the best model. Save as
`actual_vs_predicted.png` at 300 DPI.

**4.3 Feature Importance** *(code)*
Plot top-15 `feature_importances_` for the best model. Save as `feature_importance.png` at
300 DPI.

#### 5. Discussion *(markdown stub)*
Bullet prompts the student fills in: comparison narrative, where each model wins/loses,
limitations (no exogenous regressors like temperature), suggested future work (Prophet,
ARIMA baselines, deep models, weather data fusion).

#### 6. Conclusion + References *(markdown stub)*

## Critical files

- **Create:** `c:\Users\talha.yilmaz\Desktop\Talha\ds\energy_forecasting.ipynb`
- **Read (input):** `c:\Users\talha.yilmaz\Desktop\Talha\ds\DAYTON_hourly.csv`
- **Generated outputs:** `actual_vs_predicted.png`, `feature_importance.png` in project root

No existing utilities to reuse — project currently contains only raw CSVs, `.gitignore`,
README stub, and tooling configs.

## Verification

After the notebook is generated and the student executes it top-to-bottom:

1. **Runs cleanly end-to-end** — no exceptions; the printed tuning logs show all three
   `RandomizedSearchCV` runs converged to a `best_score_`.
2. **Split correctness** — printed train/val/test date ranges are strictly non-overlapping
   and in temporal order. Train ends before val starts; val ends before test starts.
3. **CV correctness** — `TimeSeriesSplit(n_splits=5)` was used (not `KFold`). Best CV RMSE
   for each model is finite and in the same order of magnitude as the val/test RMSE.
4. **Reasonableness of metrics** — DAYTON load is roughly 1.5–4 GW. With lag features
   present, expect MAE in the ~80–150 MW range, MAPE in the low single digits, R² > 0.95
   for the gradient-boosted models. R² ≈ 1.0 or MAPE < 0.5 % is a leakage warning — usually
   from forgetting `.shift(1)` on rolling features or from skipping the chronological sort.
5. **Visual checks** — `actual_vs_predicted.png` should show predicted line tracking daily
   and weekly seasonality; large systematic offsets indicate a problem.
   `feature_importance.png` should show `lag_24`, `lag_168`, and one of the `hour`
   encodings near the top — pure date features dominating with lags near zero would
   suggest the lag construction is misaligned.
