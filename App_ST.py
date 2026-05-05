import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Page config
st.set_page_config(page_title="Sales Forecast", page_icon="📦", layout="wide")

# ── Load artifacts


@st.cache_resource
def load_pipeline():
    return joblib.load("ridge_pipeline.pkl")


@st.cache_data
def load_data():
    df = pd.read_pickle("datasets/filtered_sales.pkl")
    holidays = pd.read_csv(
        "datasets/holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv("datasets/oil.csv",             parse_dates=["date"])
    transacts = pd.read_csv("datasets/transactions.csv",
                            parse_dates=["date"])
    return df, holidays, oil, transacts


pipeline = load_pipeline()
FEATURES = pipeline.feature_names_in_.tolist()
df_raw, holidays, oil, transacts = load_data()

# ── Constants
STORE_ID = 44
ITEM_ID = 1047679
# FEATURES = [
#    'y_lag1', 'y_lag2', 'y_lag7', 'y_lag14', 'y_lag21', 'y_lag28',
#    'y_roll_weekday',
#    'transactions_roll7', 'transactions_lag1', 'transactions_lag7',
#    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
#    'week_of_year', 'is_weekend', 'day_of_month',
#    'days_to_holiday', 'oil_roll7'
# ]

holiday_dates = sorted(
    holidays[holidays['type'].isin(
        ['Holiday', 'Transfer', 'Bridge'])]['date'].tolist()
)

# ── Prepare history (last 28 days of clean actuals)


@st.cache_data
def prepare_history():
    df = df_raw.copy()
    df = df[df['unit_sales'] >= 10].copy()
    # df = df.sort_values('date').reset_index(drop=True)

    # merge oil & transactions
    df = df.merge(oil[['date', 'dcoilwtico']], on='date', how='left')
    df = df.merge(
        transacts[transacts['store_nbr'] ==
                  STORE_ID][['date', 'transactions']],
        on='date', how='left'
    )
    df['dcoilwtico'] = df['dcoilwtico'].ffill()
    df['transactions'] = df['transactions'].ffill()
    df['day_of_week'] = df['date'].dt.dayofweek
    return df


history_df = prepare_history()

# ── Recursive forecast function


def forecast_recursive(n_days, safety_pct):
    history = history_df.tail(60).copy().reset_index(drop=True)
    oil_value = history['dcoilwtico'].rolling(7).mean().iloc[-1]
    predictions = []

    for _ in range(n_days):
        next_date = history['date'].iloc[-1] + pd.Timedelta(days=1)
        dow = next_date.dayofweek

        row = {
            'y_lag1':  history['unit_sales'].iloc[-1],
            'y_lag2':  history['unit_sales'].iloc[-2],
            'y_lag7':  history['unit_sales'].iloc[-7],
            'y_lag14': history['unit_sales'].iloc[-14],
            'y_lag21': history['unit_sales'].iloc[-21],
            'y_lag28': history['unit_sales'].iloc[-28],

            'y_roll_weekday': (
                history[history['day_of_week'] == dow]['unit_sales']
                .iloc[-4:].mean()
            ),

            'transactions_roll7': history['transactions'].iloc[-7:].mean(),
            'transactions_lag1':  history['transactions'].iloc[-1],
            'transactions_lag7':  history['transactions'].iloc[-7],

            'dow_sin':      np.sin(2 * np.pi * dow / 7),
            'dow_cos':      np.cos(2 * np.pi * dow / 7),
            'month_sin':    np.sin(2 * np.pi * next_date.month / 12),
            'month_cos':    np.cos(2 * np.pi * next_date.month / 12),
            'week_of_year': int(next_date.isocalendar()[1]),
            'is_weekend':   int(dow in [5, 6]),
            'day_of_month': next_date.day,
            'days_to_holiday': min(
                abs((next_date.date() - h.date()).days)
                for h in holiday_dates
            ),
            'oil_roll7': oil_value,
        }

        # --- FIXED BLOCK ---
        X = pd.DataFrame([row])
        X = X.reindex(columns=pipeline.feature_names_in_, fill_value=0)
        pred = float(np.clip(pipeline.predict(X)[0], 0, None))
        # --------------------

        suggested = round(pred * (1 + safety_pct / 100))

        if row['is_weekend']:
            day_type = "🟡 Weekend"
        elif row['days_to_holiday'] <= 2:
            day_type = "🔴 Near Holiday"
        else:
            day_type = "🟢 Normal"

        predictions.append({
            'date':            next_date,
            'predicted_sales': round(pred),
            'suggested_order': suggested,
            'day_type':        day_type,
        })

        new_row = history.iloc[-1].copy()
        new_row['date'] = next_date
        new_row['unit_sales'] = pred
        new_row['day_of_week'] = dow
        history = pd.concat(
            [history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(predictions)


# ── Sidebar
st.sidebar.title("⚙️ Settings")
n_days = st.sidebar.slider("Forecast horizon (days)", 7, 28, 7)
safety_pct = st.sidebar.slider("Safety stock buffer (%)", 0, 30, 10)
show_table = st.sidebar.checkbox("Show detailed table", value=True)
show_actual = st.sidebar.slider("Days of actual history to show", 7, 30, 14)

# ── Header
st.title("📦 Sales Forecast Dashboard")
st.markdown(
    f"**Store {STORE_ID} · Item {ITEM_ID}** — "
    "Ridge regression forecast to support weekly stock ordering."
)

# ── Run forecast
forecast_df = forecast_recursive(n_days, safety_pct)

# ── KPI cards
col1, col2, col3 = st.columns(3)
col1.metric("Total predicted sales",
            f"{forecast_df['predicted_sales'].sum():,} units")
col2.metric("Total suggested order",
            f"{forecast_df['suggested_order'].sum():,} units")
peak_date = forecast_df.loc[forecast_df['predicted_sales'].idxmax(), 'date']
col3.metric(
    "Peak day forecast",
    f"{forecast_df['predicted_sales'].max():,} units",
    peak_date.strftime("%a %d %b"),
)

st.markdown("---")

# ── Chart
actual_tail = history_df.tail(show_actual)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(actual_tail['date'], actual_tail['unit_sales'],
        color='steelblue', label='Actual sales', linewidth=2)
ax.plot(
    forecast_df['date'],
    forecast_df['predicted_sales'],
    color='darkorange',
    linestyle='--',
    linewidth=2,
    label='Forecast',
)
ax.fill_between(
    forecast_df['date'],
    forecast_df['predicted_sales'] * 0.85,
    forecast_df['predicted_sales'] * 1.15,
    color='darkorange',
    alpha=0.15,
    label='±15% band',
)
ax.axvline(history_df['date'].iloc[-1],
           color='gray', linestyle=':', linewidth=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
ax.set_ylabel("Units")
ax.legend()
ax.set_title(f"Sales Forecast — Next {n_days} Days")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ── Table
if show_table:
    st.subheader("📋 Daily Forecast Breakdown")
    display_df = forecast_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%A, %d %b %Y')
    display_df.columns = ['Date', 'Predicted Sales',
                          'Suggested Order', 'Day Type']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Footer note
st.caption(
    f"Model: Ridge Regression (scaled) · "
    f"Safety buffer applied to suggested order · "
    f"Forecast generated from last known data point: "
    f"{history_df['date'].iloc[-1].strftime('%d %b %Y')}"
)
