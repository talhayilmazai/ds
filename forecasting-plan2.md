# Plan: Add Prophet + LSTM Forecasting Paradigms to `energy_forecasting.ipynb`

## Context

The capstone notebook currently compares three tree-based regressors (Random
Forest, XGBoost, LightGBM) for Dayton hourly energy forecasting. The IEEE
report's *Methods* and *Discussion* sections are stronger when the comparison
spans **multiple forecasting paradigms**, not just multiple tree variants. The
brief from DSAI 302 also asks the student to demonstrate awareness of
statistical and deep-learning approaches.

This update appends two paradigms to the existing pipeline:

1. **Facebook Prophet** — a statistical/decomposition baseline (trend +
   multi-seasonality + holidays). It is *not* expected to beat the boosted
   trees with lag features, but it gives an honest reference for what a
   "no-feature-engineering, off-the-shelf seasonality model" achieves on the
   same horizon.
2. **PyTorch LSTM** — a deep-learning baseline that consumes the same 24-hour
   recent history the lag features encode, plus key cyclical calendar
   features. This shows whether a sequence model can match or beat the trees
   on this dataset (typically: it does not, given the strong tabular signal).

Both models are evaluated on the **identical chronological test set** used
for the trees, so the headline `results` DataFrame becomes a fair 5-model
comparison.

## Existing notebook anchors (do not change)

| Concept | Variable | Defined in cell |
|---|---|---|
| Engineered feature dataframe (Datetime is a column, not index) | `df_feat` | Cell 19 |
| Train/val/test split frames | `train_df`, `val_df`, `test_df` | Cell 22 |
| Tree-model feature columns (17 features) | `FEATURE_COLS` | Cell 22 |
| Train/val/test feature matrices and targets | `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` | Cell 22 |
| Random seed | `RANDOM_STATE = 42` | Cell 5 |
| Final fitted tree models | `rf_model`, `xgb_model`, `lgb_model` | Cell 34 |
| Comparison results table | `results` | Cell 37 |
| Best-of-trees handle | `best_model`, `best_name` | Cell 38 |

Decisions confirmed with the user:

- **DL framework:** PyTorch.
- **LSTM window:** 24-hour multivariate. Per-timestep features:
  `DAYTON_MW`, `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`,
  `is_weekend` (6 features, one of which is the target itself; standard
  practice for autoregressive sequence models).
- **Prophet:** include US federal holiday regressors via
  `add_country_holidays(country_name='US')`. Daily, weekly, yearly seasonality
  enabled.
- **Plots:** produce all three at 300 DPI.

## Changes to the notebook

The new content is inserted as **two new sub-sections** under §3, plus
**three additions** under §4. The existing §3.1–§3.4 (tree tuning) is left
untouched.

### §2.1 Setup — add imports

Append to the existing imports cell (Cell 5):

```python
# Prophet (statistical baseline)
from prophet import Prophet

# PyTorch (LSTM baseline)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Reproducibility for torch
torch.manual_seed(RANDOM_STATE)
```

If these libraries are not installed, the very first install cell of the
notebook (or a one-time `%pip install prophet torch` cell) handles that.
Prophet on Windows occasionally needs `cmdstanpy`; `pip install prophet`
pulls a working backend on current versions.

### §3.5 Statistical Baseline — Prophet *(new section)*

**Markdown cell — "3.5 Why Prophet?"**
Brief rationale: Prophet decomposes the series as `y(t) = g(t) + s(t) + h(t) + ε`
where `g` is trend, `s` is multi-period seasonality, `h` is holiday effects.
It is robust, requires no manual lag/feature engineering, and serves as a
"feature-engineering-free" statistical reference. State explicitly: we expect
it to be **less accurate** than the gradient-boosted trees on this
benchmark because Prophet does not consume lag features, but it gives a
faithful baseline for the report's comparison.

**Code cell — "Prepare Prophet input"**
```python
prophet_train = train_df[['Datetime', 'DAYTON_MW']].rename(
    columns={'Datetime': 'ds', 'DAYTON_MW': 'y'}
)
prophet_test_index = test_df[['Datetime']].rename(columns={'Datetime': 'ds'})
```

**Code cell — "Fit Prophet"**
```python
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
)
prophet_model.add_country_holidays(country_name='US')
prophet_model.fit(prophet_train)
```

Note: Prophet's fit on ~85k hourly rows takes 1–5 minutes on a laptop. This
is expected and worth a one-line markdown note so the student doesn't
interrupt the run.

**Code cell — "Predict over the test horizon"**
```python
prophet_forecast = prophet_model.predict(prophet_test_index)
y_pred_prophet = prophet_forecast['yhat'].values  # aligned to test_df ordering
```

### §3.6 Deep-Learning Baseline — PyTorch LSTM *(new section)*

**Markdown cell — "3.6 Why an LSTM?"**
Two sentences on the motivation: LSTMs explicitly model temporal
dependence through a recurrent hidden state, so they can in principle
capture the same structure that the explicit `lag_24`/`lag_168` features
encode for the trees. Including one in the comparison shows the student
considered the dominant deep-learning baseline for univariate-ish time
series.

**Code cell — "LSTM feature selection and scaling"**
```python
LSTM_FEATURES = ['DAYTON_MW', 'hour_sin', 'hour_cos',
                 'dayofweek_sin', 'dayofweek_cos', 'is_weekend']
SEQ_LEN = 24

# Fit scaler on TRAIN ONLY to avoid leakage; reuse for val and test.
feat_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_feat = feat_scaler.fit_transform(train_df[LSTM_FEATURES].values)
val_feat   = feat_scaler.transform(val_df[LSTM_FEATURES].values)
test_feat  = feat_scaler.transform(test_df[LSTM_FEATURES].values)

target_scaler.fit(train_df[['DAYTON_MW']].values)
```

**Code cell — "Build sliding-window 3D tensors"**

Helper function `make_sequences(features_2d, target_1d, seq_len)` returning
`(X_3d, y_1d)` of shape `(n_samples, seq_len, n_features)` and `(n_samples,)`.

Important detail: when building val and test sequences, we **prepend the
last `seq_len-1` rows of the previous split** so the first prediction in
val/test still has a full 24-hour history. Without this, the LSTM cannot
predict the very first val/test rows, which would silently shorten the
test set and break alignment with `y_test` and Prophet.

```python
def make_sequences(features_2d, target_1d, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(features_2d)):
        Xs.append(features_2d[i - seq_len:i])
        ys.append(target_1d[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

# Build with bridging context so val/test sequences are full length
def build_split_sequences():
    train_y_scaled = target_scaler.transform(train_df[['DAYTON_MW']]).ravel()
    val_y_scaled   = target_scaler.transform(val_df[['DAYTON_MW']]).ravel()
    test_y_scaled  = target_scaler.transform(test_df[['DAYTON_MW']]).ravel()

    # Train uses its own data only.
    X_tr, y_tr = make_sequences(train_feat, train_y_scaled, SEQ_LEN)

    # Val: prepend last SEQ_LEN train rows so first val prediction is valid.
    val_feat_ctx = np.vstack([train_feat[-SEQ_LEN:], val_feat])
    val_y_ctx    = np.concatenate([train_y_scaled[-SEQ_LEN:], val_y_scaled])
    X_va, y_va = make_sequences(val_feat_ctx, val_y_ctx, SEQ_LEN)
    # Now len(X_va) == len(val_df), aligned with y_val.

    # Test: same trick, prepend last SEQ_LEN val rows.
    test_feat_ctx = np.vstack([val_feat[-SEQ_LEN:], test_feat])
    test_y_ctx    = np.concatenate([val_y_scaled[-SEQ_LEN:], test_y_scaled])
    X_te, y_te = make_sequences(test_feat_ctx, test_y_ctx, SEQ_LEN)
    return X_tr, y_tr, X_va, y_va, X_te, y_te

X_tr_lstm, y_tr_lstm, X_va_lstm, y_va_lstm, X_te_lstm, y_te_lstm = build_split_sequences()
```

Quick assert in the cell: `assert len(X_te_lstm) == len(y_test)` — guards
against alignment regressions.

**Code cell — "Define LSTM architecture"**
```python
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # last timestep -> scalar
```

**Code cell — "Train with early stopping on val"**

A short training loop with manual early stopping (no extra dependency).
Adam optimizer, MSE loss, `patience=5` on val RMSE.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_lstm = LSTMRegressor(n_features=len(LSTM_FEATURES)).to(device)
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr_lstm), torch.from_numpy(y_tr_lstm)),
    batch_size=256, shuffle=True,
)
X_va_t = torch.from_numpy(X_va_lstm).to(device)
y_va_t = torch.from_numpy(y_va_lstm).to(device)

best_val = float('inf')
best_state = None
patience, bad_epochs = 5, 0
MAX_EPOCHS = 30

for epoch in range(MAX_EPOCHS):
    model_lstm.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model_lstm(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

    model_lstm.eval()
    with torch.no_grad():
        val_pred = model_lstm(X_va_t)
        val_rmse = torch.sqrt(loss_fn(val_pred, y_va_t)).item()

    print(f"Epoch {epoch+1:02d}  val_rmse(scaled)={val_rmse:.5f}")
    if val_rmse < best_val - 1e-5:
        best_val, best_state, bad_epochs = val_rmse, {k: v.detach().cpu().clone() for k, v in model_lstm.state_dict().items()}, 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model_lstm.load_state_dict(best_state)
```

**Code cell — "Predict and inverse-transform to MW scale"**
```python
model_lstm.eval()
with torch.no_grad():
    X_te_t = torch.from_numpy(X_te_lstm).to(device)
    y_pred_lstm_scaled = model_lstm(X_te_t).cpu().numpy().reshape(-1, 1)

y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled).ravel()
assert y_pred_lstm.shape == y_test.shape
```

### §4.1 — Append Prophet & LSTM rows to `results`

Modify Cell 37 (or add a new cell right after) so the results table covers
five models. Keep the existing `evaluate(name, model, X, y)` helper for the
trees, and add a thin variant that takes prediction arrays directly:

```python
def evaluate_preds(name, y_true, y_pred):
    return {
        'Model': name,
        'MAE':  mean_absolute_error(y_true, y_pred),
        'MAPE_%': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MSE':  mean_squared_error(y_true, y_pred),
        'R2':   r2_score(y_true, y_pred),
    }

# Append the two new rows
results = pd.concat([
    results,
    pd.DataFrame([
        evaluate_preds('Prophet', y_test.values, y_pred_prophet),
        evaluate_preds('LSTM',    y_test.values, y_pred_lstm),
    ]),
], ignore_index=True).sort_values('RMSE').reset_index(drop=True)
results
```

This keeps a single `results` DataFrame as the report's headline table.

### §4.2 — Update `actual_vs_predicted.png` to use the new global best

Cell 38's `best_model`/`best_name` selection currently runs *before* Prophet
and LSTM are added. Move the "best model" selection cell (or duplicate it)
to run *after* the appended rows so the saved
`actual_vs_predicted.png` reflects the global best across all 5 models.

For Prophet/LSTM (which aren't sklearn estimators), store predictions in a
small dict so the plotting code can pull them by name:

```python
test_preds = {
    'RandomForest': rf_model.predict(X_test),
    'XGBoost':      xgb_model.predict(X_test),
    'LightGBM':     lgb_model.predict(X_test),
    'Prophet':      y_pred_prophet,
    'LSTM':         y_pred_lstm,
}
best_name = results.iloc[0]['Model']
y_pred_best = test_preds[best_name]
```

Then the existing 14-day-tail plot consumes `y_pred_best` instead of
`best_model.predict(...)`. Save as `actual_vs_predicted.png` at 300 DPI
(unchanged filename).

### §4.4 — New plot: `models_comparison_last14d.png`

A new code cell after §4.2 that overlays Actual + all 5 model predictions on
the last 14 days of the test set (336 hours). Distinct linestyle/color per
model; Actual rendered in black with `linewidth=2`, predictions slightly
thinner. Single panel, `figsize=(12, 5)`, legend top-right, saved at 300 DPI
to `models_comparison_last14d.png`.

### §4.5 — New plot: `per_paradigm_comparison.png`

Three vertically stacked subplots (`figsize=(12, 9)`), each showing the same
14-day window:

1. **Best tree-based vs Actual** — pick best of {RF, XGB, LGBM} by RMSE
   (this can be read off `results` filtered to those three names).
2. **Prophet vs Actual**.
3. **LSTM vs Actual**.

Shared x-axis, individual titles, legend on each panel. Saved at 300 DPI to
`per_paradigm_comparison.png`. This is the figure the *Discussion* section
of the report leans on to argue per-paradigm strengths and weaknesses.

### §4.3 Feature Importance — guard against non-tree best model

If the global best model is Prophet or LSTM, `feature_importances_` does
not exist. Wrap the existing feature-importance cell in a check:

```python
if best_name in {'RandomForest', 'XGBoost', 'LightGBM'}:
    # existing code unchanged
else:
    # plot importance for the best *tree-based* model instead, with a
    # markdown note that paradigm comparison takes the place of importance
    # for the non-tree winner.
```

This avoids a crash if Prophet or LSTM happens to win and keeps the
feature-importance figure useful for the report.

## Critical files

- **Edit:** `c:\Users\talha.yilmaz\Desktop\Talha\ds\energy_forecasting.ipynb`
- **Generated outputs (300 DPI, project root):**
  - `actual_vs_predicted.png` *(updated to reflect global best across 5 models)*
  - `models_comparison_last14d.png` *(new)*
  - `per_paradigm_comparison.png` *(new)*
  - `feature_importance.png` *(unchanged behavior, now guarded)*

No new files are created. Two new package dependencies: `prophet`, `torch`.

## Verification

After running the updated notebook top-to-bottom:

1. **End-to-end success** — no exceptions; Prophet's fit log shows it
   converged; the LSTM training loop prints a per-epoch val RMSE that
   trends down and triggers early stopping (or completes 30 epochs).
2. **Alignment guards pass** — `assert len(X_te_lstm) == len(y_test)` and
   `y_pred_lstm.shape == y_test.shape` both succeed. `y_pred_prophet`
   has length `len(test_df)`.
3. **No leakage** —
   `feat_scaler` and `target_scaler` are fit on `train_df` only. Prophet is
   fit on `prophet_train` only. The val set is used solely for
   early-stopping signal (LSTM) and is never concatenated into the Prophet
   training frame.
4. **Results table coverage** — final `results` DataFrame has exactly 5 rows
   with names {RandomForest, XGBoost, LightGBM, Prophet, LSTM}, sorted by
   RMSE ascending.
5. **Reasonableness of metrics**:
   - LightGBM/XGBoost: MAE ~80–150 MW, MAPE ~3–5 %, R² > 0.95
     (unchanged from prior baseline).
   - Prophet: MAE ~150–300 MW, MAPE ~6–12 %, R² ~0.80–0.92.
     Prophet should be visibly worse than the trees — that is the
     expected, honest result.
   - LSTM: MAE ~100–200 MW, MAPE ~4–7 %, R² ~0.93–0.97. Typically
     between Prophet and the trees.
   - Any model with R² ≈ 1.0 or MAPE < 0.5 % is a leakage warning.
6. **Plots** —
   - `actual_vs_predicted.png` shows the global-best model tracking the
     last 14 days closely.
   - `models_comparison_last14d.png` shows visible spread between the 5
     models, with Prophet typically the loosest tracker and LightGBM the
     tightest.
   - `per_paradigm_comparison.png` has three readable panels with
     labeled axes, legends, and 300 DPI rendering.
