import math

from src.crop_yield_predictor import (
    benchmark_models,
    calculate_prediction_confidence,
    get_crop_suggestions,
    get_districts_for_state,
    get_historical_yield_trend,
    get_reference_values,
    get_regions,
    get_soil_types_for_state,
    get_states_for_region,
    predict_crop_yield,
    state_district_map,
    train_model,
)
from src.gemini_service import build_agronomy_context, generate_agronomy_adjustments
from app import build_feature_cards


def test_model_predicts_reasonable_yield():
    model = train_model()
    value = predict_crop_yield(
        model,
        Region="North",
        Soil_Type="Loam",
        Crop="Rice",
        Rainfall_mm=900,
        Temperature_Celsius=28,
        Fertilizer_Used=True,
        Irrigation_Used=True,
        Weather_Condition="Sunny",
        Days_to_Harvest=120,
    )

    assert isinstance(value, float)
    assert math.isfinite(value)
    assert 0 < value < 25


def test_model_evaluation_and_prediction_confidence_are_available():
    model = train_model()
    metrics = model.evaluate()
    confidence = calculate_prediction_confidence(
        model,
        Region="North",
        Soil_Type="Loam",
        Crop="Rice",
        Rainfall_mm=900,
        Temperature_Celsius=28,
        Fertilizer_Used=True,
        Irrigation_Used=True,
        Weather_Condition="Sunny",
        Days_to_Harvest=120,
    )

    assert set(metrics) >= {"rmse", "mae", "r2"}
    assert 0 <= metrics["rmse"]
    assert 0 <= confidence["score"] <= 100
    assert confidence["label"] in {"High", "Moderate", "Low"}


def test_model_benchmark_returns_multi_model_comparison():
    benchmark = benchmark_models()

    assert len(benchmark) >= 3
    assert benchmark[0]["model"] in {"RandomForestRegressor", "GradientBoostingRegressor", "LinearRegression"}
    assert set(benchmark[0]) >= {"model", "rmse", "mae", "r2"}
    assert benchmark[0]["r2"] <= 1.0
    assert benchmark[0]["rmse"] >= 0


def test_reference_values_are_state_and_crop_aware():
    reference = get_reference_values("Punjab", "Rice")
    assert reference["Region"] == "North"
    assert 400 < reference["Rainfall_mm"] < 700
    assert 20 < reference["Temperature_Celsius"] < 35


def test_state_crop_suggestions_are_rich_and_demand_oriented():
    crops = get_crop_suggestions("Punjab")
    assert len(crops) == 5
    assert crops == ["Wheat", "Rice", "Maize", "Cotton", "Barley"]
    assert "Wheat" in crops
    assert "Rice" in crops


def test_state_district_mapping_uses_all_28_states_and_realistic_climate():
    assert len(state_district_map) >= 28
    assert "Punjab" in state_district_map
    districts = get_districts_for_state("Punjab")
    assert "Amritsar" in districts
    reference = get_reference_values("Punjab", "Amritsar", "Rice")
    assert reference["District"] == "Amritsar"
    assert 500 < reference["Rainfall_mm"] < 800
    assert 20 < reference["Temperature_Celsius"] < 35


def test_region_classification_and_soil_choices_are_available():
    states = ["Punjab", "Haryana", "Uttar Pradesh"]
    north_states = get_states_for_region("North")
    assert "North" in get_regions()
    assert all(state in north_states for state in states)
    assert "Sandy Loam" in get_soil_types_for_state("Punjab", "Amritsar")
    assert "Red Soil" in get_soil_types_for_state("Tamil Nadu", "Chennai")


def test_state_district_mapping_is_substantially_expanded():
    assert len(state_district_map) >= 28
    assert min(len(districts) for districts in state_district_map.values()) >= 10
    assert "Anantapur" in state_district_map["Andhra Pradesh"]
    assert "Nagpur" in state_district_map["Maharashtra"]
    assert "Dehradun" in state_district_map["Uttarakhand"]


def test_historical_yield_trend_is_district_and_crop_aware():
    trend = get_historical_yield_trend("Punjab", "Amritsar", "Rice", "Kharif")
    assert set(trend) == {"last_year", "two_years_ago", "three_years_ago"}
    assert trend["last_year"] > trend["two_years_ago"] > trend["three_years_ago"]
    assert 2.0 < trend["last_year"] < 10.0


def test_kharif_and_rabi_profiles_are_season_aware():
    kharif = get_reference_values("Punjab", "Amritsar", "Rice", "Kharif")
    rabi = get_reference_values("Punjab", "Amritsar", "Rice", "Rabi")

    assert kharif["Rainfall_mm"] > rabi["Rainfall_mm"]
    assert kharif["Temperature_Celsius"] > rabi["Temperature_Celsius"]


def test_agronomy_guidance_changes_with_crop_soil_and_season():
    rice_loam = generate_agronomy_adjustments("Punjab", "Amritsar", "Loam", "Rice", 700, 28, "Kharif")
    rice_clay = generate_agronomy_adjustments("Punjab", "Amritsar", "Clay", "Rice", 700, 28, "Kharif")
    wheat_rabi = generate_agronomy_adjustments("Punjab", "Amritsar", "Loam", "Wheat", 450, 18, "Rabi")

    assert rice_loam["nutrient_recommendation"] != rice_clay["nutrient_recommendation"]
    assert rice_loam["seasonal_actions"] != rice_clay["seasonal_actions"]
    assert rice_loam["nutrient_recommendation"] != wheat_rabi["nutrient_recommendation"]
    assert rice_loam["seasonal_actions"] != wheat_rabi["seasonal_actions"]


def test_feature_cards_adjust_by_crop_soil_and_season():
    kharif_cards = build_feature_cards("Punjab", "Amritsar", "Rice", "Kharif", "Loam", 700, 28)
    rabi_cards = build_feature_cards("Punjab", "Amritsar", "Rice", "Rabi", "Sandy", 420, 19)
    clay_cards = build_feature_cards("Punjab", "Amritsar", "Rice", "Kharif", "Clay", 700, 28)

    assert kharif_cards[0]["title"] != rabi_cards[0]["title"]
    assert kharif_cards[2]["title"] != clay_cards[2]["title"]
    assert kharif_cards[3]["title"] != rabi_cards[3]["title"]


def test_agronomy_context_includes_all_variable_factors():
    summary = build_agronomy_context(
        state="Punjab",
        district="Amritsar",
        soil_type="Clay",
        crop="Rice",
        rainfall=700,
        temperature=28,
        season="Kharif",
        weather_condition="Rainy",
        fertilizer_used=True,
        irrigation_used=True,
        days_to_harvest=120,
    )

    assert summary["state"] == "Punjab"
    assert summary["district"] == "Amritsar"
    assert summary["soil_type"] == "Clay"
    assert summary["crop"] == "Rice"
    assert summary["season"] == "Kharif"
    assert "rainfall_mm" in summary
    assert "temperature_c" in summary
    assert "weather_condition" in summary
    assert "nutrient_need" in summary
    assert "risk_summary" in summary


def test_prediction_response_includes_historical_yield_summary():
    from app import app

    client = app.test_client()
    response = client.post(
        "/predict",
        data={
            "Region": "North",
            "State": "Punjab",
            "District": "Amritsar",
            "Crop": "Rice",
            "Season": "Kharif",
            "Soil_Type": "Loam",
            "Rainfall_mm": "700",
            "Temperature_Celsius": "28",
            "Days_to_Harvest": "120",
            "Weather_Condition": "Sunny",
            "Fertilizer_Used": "True",
            "Irrigation_Used": "True",
        },
    )

    assert response.status_code == 200
    assert b"yield_last_year" in response.data or b"Previous year yield trend in this area" in response.data
    assert b"Last year" in response.data
