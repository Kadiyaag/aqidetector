import pandas as pd
import numpy as np

np.random.seed(42)

samples = 5000

data = {
    "CO": np.random.uniform(0.1, 5, samples),
    "CO2": np.random.uniform(350, 2000, samples),   # new feature
    "PM25": np.random.uniform(5, 300, samples),
    "PM10": np.random.uniform(10, 400, samples),
    "NO2": np.random.uniform(5, 200, samples)
}

df = pd.DataFrame(data)

# synthetic AQI formula
df["AQI"] = (
    df["PM25"] * 0.5 +
    df["PM10"] * 0.2 +
    df["CO"] * 15 +
    df["NO2"] * 0.3 +
    df["CO2"] * 0.02
)

df["AQI"] = df["AQI"].astype(int)

df.to_csv("data/aqi_dataset.csv", index=False)

print("Dataset generated successfully")