# %% [markdown]
# # Task 5 (Advanced) – Netflix Trend Forecasting Model
# Forecast how many titles Netflix adds per month. The target is the monthly count of titles by `date_added`
# (the only true time stamp in the data; `release_year` is when a title was *made*, not when it reached Netflix).
# The data ends in Sep 2021, so we forecast Oct 2021 – Sep 2022.

# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from common import load_clean, save_fig

df = load_clean()
monthly_all = df.set_index("date_added").resample("MS").size()
# Before 2016 there are only a handful of titles per month (catalogue still being built / sampled) -> start in 2016
y = monthly_all["2016-01":].astype(float)
print(f"{len(y)} months: {y.index.min():%b %Y} -> {y.index.max():%b %Y}")

# %% [markdown]
# ## Step 1 – Prepare time-based features
# Time index, month-of-year (one-hot), and lag features (1, 2, 3, 12 months and a 3-month rolling mean).

# %%
def calendar_features(idx, t0):
    X = pd.DataFrame(index=idx)
    X["t"] = ((idx.year - t0.year) * 12 + idx.month - t0.month).astype(float)
    for m in range(1, 13):
        X[f"m{m}"] = (idx.month == m).astype(float)
    return X

def lag_features(series, idx):
    """Features for the months in idx using only values known *before* each month (shifted)."""
    s = series
    X = calendar_features(idx, y.index[0])[["t"]].copy()
    for L in (1, 2, 3, 12):
        X[f"lag{L}"] = s.shift(L).reindex(idx)
    X["roll3"] = s.shift(1).rolling(3).mean().reindex(idx)
    X["month"] = idx.month
    return X

print(calendar_features(y.index, y.index[0]).head(3).iloc[:, :6])

# %% [markdown]
# ## Step 2 – Analyze historical release patterns

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
ax[0].plot(y.index, y.values, c="#e50914"); ax[0].plot(y.index, y.rolling(12).mean(), c="#221f1f", lw=2, label="12-month average")
ax[0].set_title("Titles added to Netflix per month"); ax[0].legend()
season = y["2018":"2020"].groupby(y["2018":"2020"].index.month).mean()
ax[1].bar(season.index, season.values, color="#e50914"); ax[1].set_xticks(range(1, 13))
ax[1].set_title("Average titles added by calendar month (2018-2020, full years)"); ax[1].set_xlabel("month")
save_fig("task5_history_and_seasonality.png")

yearly = df.groupby("year_added").size().loc[2016:]
print(yearly.to_string())
print("\nBusiest month (2018-20 avg):", season.idxmax(), "| quietest:", season.idxmin())
by_type = df[df["year_added"] >= 2016].groupby(["year_added", "type"]).size().unstack()
print(by_type.assign(tv_share=(by_type["TV Show"] / by_type.sum(axis=1)).round(3)).to_string())

# %% [markdown]
# ## Step 3 – Build forecasting models
# Time-ordered split: **train = Jan 2016 – Sep 2020, test = last 12 months (Oct 2020 – Sep 2021)**.
# Never shuffle a time series. Models:
# 1. Seasonal naive (same month last year) – baseline
# 2. Linear trend + month dummies
# 3. Ridge on trend + month dummies (log scale)
# 4. Random Forest and 5. Gradient Boosting on lag features (recursive multi-step forecasting)

# %%
H = 12
train, test = y.iloc[:-H], y.iloc[-H:]

def fit_forecast(name, tr, horizon_idx):
    """Fit on series `tr`, return forecast (Series) for horizon_idx."""
    if name == "Seasonal naive":
        return pd.Series([tr.iloc[-12 + (i % 12)] for i in range(len(horizon_idx))], index=horizon_idx)
    if name == "Linear trend + month":
        m = LinearRegression().fit(calendar_features(tr.index, y.index[0]), tr)
        return pd.Series(m.predict(calendar_features(horizon_idx, y.index[0])), index=horizon_idx)
    if name == "Ridge log-trend + month":
        m = Ridge(alpha=1.0).fit(calendar_features(tr.index, y.index[0]), np.log1p(tr))
        return pd.Series(np.expm1(m.predict(calendar_features(horizon_idx, y.index[0]))), index=horizon_idx)
    # tree models on lag features, recursive forecast
    mk = {"Random Forest (lags)": lambda: RandomForestRegressor(300, min_samples_leaf=2, random_state=42),
          "Gradient Boosting (lags)": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42)}[name]
    Xtr = lag_features(tr, tr.index).dropna()
    m = mk().fit(Xtr, tr.loc[Xtr.index])
    hist = tr.copy()
    for ts in horizon_idx:
        hist.loc[ts] = np.nan
        row = lag_features(hist, pd.DatetimeIndex([ts]))
        hist.loc[ts] = float(m.predict(row)[0])
    return hist.loc[horizon_idx]

names = ["Seasonal naive", "Linear trend + month", "Ridge log-trend + month", "Random Forest (lags)", "Gradient Boosting (lags)"]
fc = {n: fit_forecast(n, train, test.index) for n in names}

# %% [markdown]
# ## Step 4 – Evaluate forecasting accuracy (hold-out + rolling-origin backtest)
# A single 12-month test can be lucky, so we also run a rolling-origin backtest: 4 origins, each forecasting the
# next 6 months from data before the origin.

# %%
def metrics(actual, pred):
    return {"MAE": mean_absolute_error(actual, pred), "RMSE": mean_squared_error(actual, pred) ** .5,
            "MAPE %": 100 * np.mean(np.abs((actual - pred) / actual))}

res = pd.DataFrame({n: metrics(test, fc[n]) for n in names}).T

roll = {n: [] for n in names}
for end in range(len(y) - 24, len(y) - 5, 6):  # origins
    tr, te = y.iloc[:end], y.iloc[end:end + 6]
    for n in names:
        roll[n].append(mean_absolute_error(te, fit_forecast(n, tr, te.index)))
res["Backtest MAE (6-mo, 4 origins)"] = {n: np.mean(v) for n, v in roll.items()}
res = res.round(2).sort_values("MAE")
print(res.to_string()); res.to_csv("outputs/task5_model_comparison.csv")
print(f"\nMean of test months = {test.mean():.1f} titles/month")

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(y.index[-36:], y.iloc[-36:], c="black", lw=2, label="Actual")
for n, c in zip(names, ["#999", "#0071eb", "#2ca02c", "#f5a623", "#e50914"]):
    ax.plot(fc[n].index, fc[n].values, "--", c=c, label=n)
ax.axvline(train.index[-1], c="grey", ls=":"); ax.set_title("Hold-out forecast: last 12 months"); ax.legend(fontsize=8, ncol=2)
save_fig("task5_holdout_forecasts.png")

# %% [markdown]
# ## Step 5 – Generate future predictions (Oct 2021 – Sep 2022)
# Refit the best model (lowest backtest MAE, i.e. judged on several origins rather than only the final window) on all data.

# %%
best = res["Backtest MAE (6-mo, 4 origins)"].idxmin()
print("Selected model:", best)
future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
future = fit_forecast(best, y, future_idx)

# rough 80% band from the model's backtest errors (empirical, not a formal interval)
resid = np.concatenate([(test - fc[best]).values])
lo, hi = future + np.percentile(resid, 10), future + np.percentile(resid, 90)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(y.index, y.values, c="#221f1f", label="History")
ax.plot(future.index, future.values, c="#e50914", lw=2, label=f"Forecast – {best}")
ax.fill_between(future.index, lo.clip(lower=0), hi, color="#e50914", alpha=.15, label="approx. 80% band")
ax.set_title("Forecast of titles added per month: Oct 2021 – Sep 2022"); ax.legend()
save_fig("task5_future_forecast.png")

out = pd.DataFrame({"forecast": future.round(0), "low": lo.clip(lower=0).round(0), "high": hi.round(0)})
print(out.to_string()); out.to_csv("outputs/task5_future_forecast.csv")
print(f"\nForecast total next 12 months: {future.sum():.0f} vs last 12 months actual: {y.iloc[-12:].sum():.0f}")

# %% [markdown]
# ## Bonus – Movies vs TV Shows trend
# Same approach per type (seasonal-trend model chosen above), to see which format is expected to grow.

# %%
fig, ax = plt.subplots(figsize=(10, 4))
for t, c in [("Movie", "#e50914"), ("TV Show", "#0071eb")]:
    s = df[df["type"] == t].set_index("date_added").resample("MS").size()["2016-01":].astype(float)
    f = fit_forecast(best, s.reindex(y.index).fillna(0), future_idx) if best != "Seasonal naive" else None
    ax.plot(s.index, s.values, c=c, alpha=.5)
    if f is not None:
        ax.plot(f.index, f.values, c=c, lw=2, label=f"{t}: forecast {f.sum():.0f} titles")
ax.set_title("Monthly titles added by type, with 12-month forecast"); ax.legend()
save_fig("task5_type_forecast.png")
