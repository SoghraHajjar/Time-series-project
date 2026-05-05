📦 Sales Forecast Dashboard
📝 Project Overview
This project provides an interactive Sales Forecast Dashboard built with Streamlit.
It predicts future sales for a specific store–item combination using a Ridge Regression model and supports data‑driven stock ordering decisions.

The dashboard helps retail managers and analysts:

anticipate demand

avoid stockouts

reduce over‑ordering

prepare for holiday‑driven spikes

plan operations based on data rather than intuition

🚀 Key Features
7–28 day sales forecast using recursive prediction

Ridge Regression model with standardized features

Automatic feature engineering, including:

lag features

rolling averages

transaction‑based features

seasonal patterns (weekday & month)

holiday proximity

oil price trends

Interactive controls for forecast horizon and safety stock

KPI summary cards

Forecast chart with ±15% uncertainty band

Daily breakdown table with day classification

Fast performance through Streamlit caching

📊 Data Sources
The model uses four combined datasets:

Filtered sales history

Holiday calendar (Holiday, Transfer, Bridge days)

Oil price data (dcoilwtico)

Store transaction counts

These datasets are merged, cleaned, and used to generate all model features.

🔧 Model & Methodology
The forecasting logic is based on a recursive Ridge Regression pipeline.

Forecasting steps:
Start with the last 60 days of historical data

Generate a feature vector for the next day:

sales lags (1, 2, 7, 14, 21, 28)

rolling weekday averages

transaction lags and rolling means

sine/cosine seasonal encodings

week of year, day of month

days to next holiday

7‑day rolling oil price

Predict the next day

Append the prediction to history

Repeat for the full forecast horizon

This approach captures short‑term patterns, seasonality, and holiday effects.

🖥️ Dashboard Structure
Sidebar Controls
Forecast horizon (7–28 days)

Safety stock buffer (%)

Toggle for detailed table

Number of historical days to display

KPI Cards
Total predicted sales

Total suggested order

Peak forecast day

Forecast Chart
Actual sales (recent days)

Predicted sales (future days)

±15% uncertainty band

Vertical line marking the transition from actuals to forecast

Daily Forecast Table
Date

Predicted sales

Suggested order

Day type (Weekend, Near Holiday, Normal)

▶️ How to Run the App
From the project directory:

bash
streamlit run App_ST.py
The dashboard will open automatically in your browser at:

Code
http://localhost:8501
📁 Project Structure
Code
├── App_ST.py                 # Streamlit dashboard
├── ridge_pipeline.pkl        # Trained Ridge Regression model
├── datasets/
│   ├── filtered_sales.pkl
│   ├── holidays_events.csv
│   ├── oil.csv
│   └── transactions.csv
└── README.md
📌 Requirements
Python 3.9+

Install dependencies:

bash
pip install streamlit pandas numpy joblib matplotlib
📄 License
This project is intended for academic and non‑commercial use.
Please cite the repository when reusing or extending the code.
