import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import timedelta

# Load data
@st.cache
def load_data():
    train_df = pd.read_csv('train.csv')
    store_df = pd.read_csv('store.csv')
    df = pd.merge(train_df, store_df, on='Store')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # Handle 'PromoInterval'
    promo_intervals = df['PromoInterval'].unique().tolist()
    promo_interval_dict = {k: v for v, k in enumerate(promo_intervals)}
    df['PromoInterval'] = df['PromoInterval'].map(promo_interval_dict)
    
    df = pd.get_dummies(df, columns=['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment'])
    return df

df = load_data()

# User input
store_id = st.sidebar.selectbox('Select Store', df['Store'].unique())
date = st.sidebar.date_input('Select Date')

# Filter data for selected store
store_data = df[df['Store'] == store_id]

# Split data into training and test sets
train_data, test_data = train_test_split(store_data, test_size=0.2, shuffle=False)

# Extend test data to include future dates
last_date = test_data.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), end=date)
future_data = pd.DataFrame(index=future_dates, columns=test_data.columns)
test_data = pd.concat([test_data, future_data])

# ARIMA Model
arima_model = ARIMA(train_data['Sales'], order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=len(test_data))

# Prepare data for Random Forest, including future dates
X_train = train_data.drop(columns=['Sales'])
y_train = train_data['Sales']

# Fill future data with appropriate values (you might need to adjust this part based on your feature set)
test_data.fillna(method='ffill', inplace=True)
X_test = test_data.drop(columns=['Sales'])
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_forecast = rf_model.predict(X_test)

# Combine forecasts
ensemble_forecast = 0.5 * arima_forecast + 0.5 * rf_forecast

# Calculate the forecast index
forecast_index = (pd.to_datetime(date) - test_data.index[0]).days
forecast_value = ensemble_forecast[forecast_index]

# Display forecast
st.write(f"Forecasted Sales for Store {store_id} on {date}: {forecast_value}")
