import pytest
from src.explainability import (
    explain_prediction,
    get_global_feature_importances,
)


def test_explain_prediction_decomposes_features():
    explanation = explain_prediction(
        predicted_yield=7.2,
        rainfall_mm=950.0,
        temperature_celsius=28.0,
        fertilizer_used=True,
        irrigation_used=True,
        crop="Rice",
        soil_type="Clay",
        days_to_harvest=120,
    )

    assert "contributions" in explanation
    assert len(explanation["contributions"]) >= 4
    assert "what_if" in explanation
    assert "key_takeaway" in explanation
    assert explanation["predicted_yield"] == 7.2

    # Check that what-if without fertilizer decreases yield
    assert explanation["what_if"]["without_fertilizer"] < explanation["predicted_yield"]
    assert explanation["what_if"]["without_irrigation"] < explanation["predicted_yield"]


def test_explain_prediction_handles_no_inputs():
    explanation = explain_prediction(
        predicted_yield=3.5,
        rainfall_mm=400.0,
        temperature_celsius=32.0,
        fertilizer_used=False,
        irrigation_used=False,
        crop="Wheat",
        soil_type="Sandy",
    )

    assert explanation["what_if"]["fertilizer_gain"] > 0
    assert explanation["what_if"]["irrigation_gain"] > 0
    assert "Both fertilizer and irrigation are inactive" in explanation["key_takeaway"]


def test_get_global_feature_importances_returns_ranked_features():
    importances = get_global_feature_importances()
    assert len(importances) >= 5
    assert importances[0]["feature"] in {"Fertilizer_Used", "Rainfall_mm"}
    assert all("importance" in item and "effect" in item for item in importances)
