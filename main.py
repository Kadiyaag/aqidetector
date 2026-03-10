from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load models
model = joblib.load("model/aqi_model.pkl")
forecast_model = joblib.load("model/aqi_forecast_model.pkl")


class SensorData(BaseModel):
    CO: float
    CO2: float
    PM25: float
    PM10: float
    NO2: float
    SpO2: float
    heart_rate: float


# AQI category
def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# Health risk (uses SpO2 + heart rate but not returned)
def get_health_risk(aqi, spo2, heart_rate):

    if spo2 < 90 or aqi > 200:
        return "Very High"
    elif spo2 < 94 or aqi > 150:
        return "High"
    elif aqi > 100:
        return "Moderate"
    else:
        return "Low"


# Forecast helper
def forecast_aqi(model, current_aqi):

    predictions = []

    lag1 = current_aqi
    lag2 = current_aqi
    lag3 = current_aqi

    for step in range(5):

        input_data = [[lag1, lag2, lag3]]

        pred = model.predict(input_data)[0]

        predictions.append(int(pred))

        lag3 = lag2
        lag2 = lag1
        lag1 = pred

    return predictions


@app.get("/")
def home():
    return {"message": "AQI Forecast + Health Risk API running"}


@app.post("/forecast")
def forecast(data: SensorData):

    # Predict current AQI
    input_data = pd.DataFrame([{
        "CO": data.CO,
        "CO2": data.CO2,
        "PM25": data.PM25,
        "PM10": data.PM10,
        "NO2": data.NO2
    }])

    current_aqi = int(model.predict(input_data)[0])

    category = get_aqi_category(current_aqi)

    health_risk = get_health_risk(current_aqi, data.SpO2, data.heart_rate)

    future_predictions = forecast_aqi(forecast_model, current_aqi)

    hours = [5, 10, 15, 20, 24]

    forecast_result = []

    for i in range(len(hours)):

        aqi_value = int(future_predictions[i])

        forecast_result.append({
            "hours_ahead": hours[i],
            "AQI": aqi_value,
            "category": get_aqi_category(aqi_value)
        })

    return {
        "current_AQI": current_aqi,
        "category": category,
        "health_risk": health_risk,
        "forecast": forecast_result
    }