import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/aqi_dataset.csv")

X = df.drop("AQI", axis=1)
y = df["AQI"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model/aqi_model.pkl")

print("Model trained and saved")