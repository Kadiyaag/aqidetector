import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# load dataset
df = pd.read_csv("data/aqi_dataset.csv")

# create lag features for forecasting
df["AQI_lag1"] = df["AQI"].shift(1)
df["AQI_lag2"] = df["AQI"].shift(2)
df["AQI_lag3"] = df["AQI"].shift(3)

# remove missing rows
df = df.dropna()

X = df[["AQI_lag1", "AQI_lag2", "AQI_lag3"]]
y = df["AQI"]

# train model
model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X, y)

# save forecast model
joblib.dump(model, "model/aqi_forecast_model.pkl")

print("Forecast model trained and saved")