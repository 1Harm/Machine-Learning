import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load dataset
df = pd.read_csv('dump_events_sep_2_weeks.csv', parse_dates=['time'])

# Ensure the 'time' column is in datetime format and the 'effect' column is a float
df['Date'] = pd.to_datetime(df['time'])
df['effect'] = df['effect'].astype(float)  # Adjusted column name to 'effect'

# Aggregate data to daily level - summing up the 'effect' column
df_daily = df.groupby(df['Date'].dt.date).agg({'effect':'sum'}).reset_index()  # Adjusted column name to 'effect'

# Rename columns to fit Prophet requirements
df_daily.columns = ['ds', 'y']

# Fit the prophet model
prophet = Prophet()
prophet.fit(df_daily)

# Create future dataframe for forecasting
future = prophet.make_future_dataframe(periods=365)

# Predict the future
forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Plot the forecast
fig = prophet.plot(forecast)
plt.show()
