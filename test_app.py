import json
from app import app


def test_home_route_renders_template():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Crop Yield Predictor" in response.data


def test_prediction_route_renders_result():
    client = app.test_client()
    response = client.post(
        "/predict",
        data={
            "State": "Punjab",
            "Region": "North",
            "Soil_Type": "Loam",
            "Crop": "Rice",
            "Rainfall_mm": "900",
            "Temperature_Celsius": "28",
            "Fertilizer_Used": "True",
            "Irrigation_Used": "True",
            "Weather_Condition": "Sunny",
            "Days_to_Harvest": "120",
        },
    )

    assert response.status_code == 200
    assert b"Predicted Yield" in response.data


def test_api_predict_json():
    client = app.test_client()
    payload = {
        "State": "Punjab",
        "Region": "North",
        "District": "Amritsar",
        "Soil_Type": "Loam",
        "Crop": "Wheat",
        "Season": "Rabi",
        "Rainfall_mm": 450,
        "Temperature_Celsius": 22,
        "Fertilizer_Used": True,
        "Irrigation_Used": True,
        "Weather_Condition": "Sunny",
        "Days_to_Harvest": 130,
        "Land_Area_Ha": 2.5,
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "predicted_yield" in data
    assert data["predicted_yield"] > 0
    assert "confidence" in data
    assert "economics" in data
    assert data["economics"]["land_area_ha"] == 2.5
    assert data["economics"]["gross_revenue_inr"] > 0
    assert "explanation" in data
    assert len(data["explanation"]["contributions"]) > 0


def test_api_models():
    client = app.test_client()
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["benchmark"]) >= 3
    assert all("model" in m and "r2" in m for m in data["benchmark"])


def test_api_feature_importance():
    client = app.test_client()
    response = client.get("/api/feature-importance")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["feature_importances"]) >= 5


def test_api_health():
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


def test_api_predict_yield_changes_with_crop():
    client = app.test_client()
    base_payload = {
        "State": "Punjab",
        "Region": "North",
        "District": "Amritsar",
        "Soil_Type": "Loam",
        "Season": "Kharif",
        "Rainfall_mm": 900,
        "Temperature_Celsius": 28,
        "Fertilizer_Used": True,
        "Irrigation_Used": True,
        "Weather_Condition": "Sunny",
        "Days_to_Harvest": 120,
        "Land_Area_Ha": 2.5,
    }

    rice = client.post("/api/predict", data=json.dumps({**base_payload, "Crop": "Rice"}), content_type="application/json")
    wheat = client.post("/api/predict", data=json.dumps({**base_payload, "Crop": "Wheat"}), content_type="application/json")

    assert rice.status_code == 200
    assert wheat.status_code == 200
    assert rice.get_json()["predicted_yield"] != wheat.get_json()["predicted_yield"]


def test_api_predict_yield_is_realistic():
    client = app.test_client()
    payload = {
        "State": "Punjab",
        "Region": "North",
        "District": "Amritsar",
        "Soil_Type": "Loam",
        "Crop": "Rice",
        "Season": "Kharif",
        "Rainfall_mm": 900,
        "Temperature_Celsius": 28,
        "Fertilizer_Used": True,
        "Irrigation_Used": True,
        "Weather_Condition": "Sunny",
        "Days_to_Harvest": 120,
        "Land_Area_Ha": 5,
    }

    response = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert 0 < data["predicted_yield"] <= 5.5
    assert data["predicted_yield"] < 5.5


def test_api_states_returns_states_for_region():
    client = app.test_client()
    response = client.get("/api/states?region=South")
    assert response.status_code == 200
    data = response.get_json()
    assert data["region"] == "South"
    assert "Andhra Pradesh" in data["states"]
    assert "Tamil Nadu" in data["states"]


def test_api_districts_returns_districts_for_state():
    client = app.test_client()
    response = client.get("/api/districts?state=Tamil%20Nadu")
    assert response.status_code == 200
    data = response.get_json()
    assert data["state"] == "Tamil Nadu"
    assert "Chennai" in data["districts"]
    assert "Coimbatore" in data["districts"]
