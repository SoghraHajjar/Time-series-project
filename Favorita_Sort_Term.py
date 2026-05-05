from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

store_ids = [44]
item_ids = [1047679]
max_date = '2014-04-01'
df_filtered = pd.read_pickle("datasets/filtered_sales.pkl")
# Flag and optionally exclude anomalous days before creating lags
# adjust threshold as needed
threshold = df_filtered['unit_sales'].quantile(0.02)
df_filtered['is_anomaly'] = df_filtered['unit_sales'] < threshold
df_clean = df_filtered[~df_filtered['is_anomaly']].copy()
# Then reindex and forward-fill the gap before computing lags
# df_clean = df_clean.set_index('date').asfreq('D').reset_index()
df_clean['unit_sales'] = df_clean['unit_sales'].fillna(method='ffill')

# series = TimeSeries.from_dataframe(df_filtered, value_cols='unit_sales')
# train, test = series.split_after(0.8)
# train.plot(title="Raw unit_sales (d=0)")
oil = pd.read_csv("datasets/oil.csv",             parse_dates=["date"])
holidays = pd.read_csv("datasets/holidays_events.csv", parse_dates=["date"])
transacts = pd.read_csv("datasets/transactions.csv",    parse_dates=["date"])
holidays_filtered = holidays[holidays['date'] < max_date]
oil_filtered = oil[oil['date'] < max_date]
transacts_filtered = transacts[transacts['store_nbr'].isin(
    store_ids) & (transacts['date'] < max_date)]
# merge the datasets on date
df_merged = df_clean.merge(
    holidays_filtered[['date', 'type']], on='date', how='left')
holiday_dates = set(holidays_filtered['date'])
holiday_dates = set(
    holidays_filtered[holidays_filtered['type'].isin(
        ['Holiday', 'Transfer', 'Bridge'])]
    ['date']
)


df_merged = df_merged.merge(
    oil_filtered[['date', 'dcoilwtico']], on='date', how='left')
df_merged = df_merged.merge(
    transacts_filtered[['date', 'transactions']], on='date', how='left')
# Fill missing values in the new features
df_merged['type'] = df_merged['type'].fillna('None')
df_merged['is_holiday'] = (df_merged['type'] != 'None').astype(int)
df_merged['dcoilwtico'] = df_merged['dcoilwtico'].fillna(method='ffill')
df_merged['transactions'] = df_merged['transactions'].shift(1)
df_merged['transactions'] = df_merged['transactions'].ffill()
# ----------------------------------------------------
df = df_merged.copy()


# --- Sales lag features ---
df['y_lag1'] = df['unit_sales'].shift(1)
df['y_lag2'] = df['unit_sales'].shift(2)
df['y_lag7'] = df['unit_sales'].shift(7)
df['y_lag14'] = df['unit_sales'].shift(14)
df['y_lag21'] = df['unit_sales'].shift(21)
df['y_lag28'] = df['unit_sales'].shift(28)


# --- Sales rolling features ---
df['y_roll7'] = df['unit_sales'].shift(1).rolling(7).mean()
df['y_roll14'] = df['unit_sales'].shift(1).rolling(14).mean()
# --- Transaction features ---
df['transactions_lag1'] = df['transactions'].shift(1)
df['transactions_lag7'] = df['transactions'].shift(7)
df['transactions_roll7'] = df['transactions'].shift(1).rolling(7).mean()
# --- Calendar features ---

df['month'] = df['date'].dt.month
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)


# Day of week — often the single biggest calendar feature for beverages
df['day_of_week'] = df['date'].dt.dayofweek
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
# Rolling same-weekday mean (last 4 weeks)
df['y_roll_weekday'] = (
    df.groupby('day_of_week')['unit_sales']
    .transform(lambda x: x.shift(1).rolling(4).mean())
)

# Is it a weekend?
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Day of month (payday effect)
df['day_of_month'] = df['date'].dt.day
#
# Holiday type encoding — 'Holiday', 'Transfer', 'Bridge', etc.
# carry different effects
df = pd.get_dummies(df, columns=['type'], drop_first=True)
# Or: days near a holiday often spike too
df['days_to_holiday'] = df['date'].apply(
    lambda d: min(abs((d - h).days) for h in holiday_dates)
).clip(upper=7)
#
# df['oil_lag1'] = df['dcoilwtico'].shift(1)
df['oil_roll7'] = df['dcoilwtico'].shift(1).rolling(7).mean()


# --- Drop rows with NaNs created by lag/rolling windows ---
df = df.dropna().reset_index(drop=True)
# --- Final feature set ---
FEATURES = [
    # Sales lags
    'y_lag1', 'y_lag2', 'y_lag7', 'y_lag14', 'y_lag21', 'y_lag28',
    # Sales rolling
    'y_roll_weekday',
    # Transactions
    'transactions_lag1',
    'transactions_lag7', 'transactions_roll7',
    # oil
    'oil_roll7',
    # Calendar
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'week_of_year',
    'is_weekend', 'day_of_month', 'days_to_holiday'
]
TARGET = 'unit_sales'
# --- Train-test split ---
split_date = pd.to_datetime('2014-01-01')
train = df[df['date'] <= split_date]
test = df[df['date'] > split_date]
X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]
# -------------------------------------------------------------
# ---------------------------models---------------------------
# -------------------------------------------------------------

models = {
    'Ridge':    Ridge(alpha=1.0),
    'LightGBM': LGBMRegressor(n_estimators=200,
                              learning_rate=0.03,
                              num_leaves=8,        # was 16 — reduce complexity
                              min_child_samples=10,
                              subsample=0.8,
                              random_state=42),
    'XGBoost':  XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,         # was 5 — shallower trees
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
}

results = {}
epsilon = 1e-5
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)
    results[name] = {
        'MAE':  mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAPE': np.mean(np.abs((y_test - preds) / (y_test + epsilon))) * 100
    }

results_df = pd.DataFrame(results).T.sort_values('RMSE')
print(results_df)

# ---------------tuning----------------

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(
        alphas=[0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_root_mean_squared_error'
    ))
])

pipeline.fit(X_train, y_train)
preds = np.clip(pipeline.predict(X_test), 0, None)


print("Best alpha:", pipeline.named_steps['ridge'].alpha_)
# preds = np.clip(model_ridge.predict(X_test), 0, None)
print("MAE: ", mean_absolute_error(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
# -----------------
plt.figure(figsize=(12, 5))
plt.plot(test['date'], y_test, label='Actual')
plt.plot(test['date'], preds, label='Predicted')
plt.legend()
plt.title('Model Performance')
plt.show()
# ---------------------------
# preds_ridge = np.clip(model_ridge.predict(X_test), 0, None)
preds_ridge = np.clip(pipeline.predict(X_test), 0, None)
residuals = y_test.values - preds_ridge

plt.figure(figsize=(12, 4))
plt.plot(test['date'], residuals, color='steelblue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Ridge Residuals — Validation Period')
plt.ylabel('Actual - Predicted')
plt.show()
# ---------------
coef_df = pd.DataFrame({
    'feature': FEATURES,
    'coefficient': pipeline.named_steps['ridge'].coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(coef_df)

final_preds = np.clip(pipeline.predict(X_test), 0, None)
print("Final Test MAE: ", mean_absolute_error(y_test, final_preds))
print("Final Test RMSE:", np.sqrt(mean_squared_error(y_test, final_preds)))
print("Final Test MAPE:", np.mean(
    np.abs((y_test - final_preds) / (y_test + epsilon))) * 100)

joblib.dump(pipeline, "ridge_pipeline.pkl")
