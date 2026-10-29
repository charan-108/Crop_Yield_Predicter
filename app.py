from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from src.crop_yield_predictor import (
    DISTRICT_CLIMATE,
    REGION_TO_STATES,
    STATE_CROP_SUGGESTIONS,
    STATE_TO_REGION,
    benchmark_models,
    calculate_prediction_confidence,
    get_crop_suggestions,
    get_districts_for_state,
    get_historical_yield_trend,
    get_reference_values,
    get_regions,
    get_region_for_state,
    get_soil_types_for_state,
    get_states_for_region,
    predict_crop_yield,
    state_district_map,
    train_model,
)
from src.economics import calculate_crop_economics
from src.explainability import explain_prediction, get_global_feature_importances
from src.gemini_service import (
    build_agronomy_context,
    generate_agronomy_adjustments,
    generate_crop_recommendation,
)
from src.soil_service import get_soil_for_district
from src.weather_service import get_weather_for_district

app = Flask(__name__)


def build_feature_cards(
    state: str,
    district: str,
    crop: str,
    season: str,
    soil_type: str,
    rainfall: float,
    temperature: float,
) -> list[dict[str, str]]:
    crop_name = (crop or "Rice").strip().title()
    season_name = (season or "Kharif").strip().title()
    district_name = (district or get_districts_for_state(state)[0]).strip().title()
    soil_name = (soil_type or "Loam").strip()
    soil_lower = soil_name.lower()
    rainfall_value = float(rainfall or 0)
    temp_value = float(temperature or 0)

    if season_name == "Kharif":
        yield_confidence = "High" if rainfall_value >= 700 and temp_value <= 32 else "Moderate"
        irrigation_plan = "Smart drip" if rainfall_value >= 600 else "Targeted drip"
        pest_risk = "Low to moderate" if rainfall_value < 1300 else "Moderate"
        action = "Topdress nutrients" if crop_name in {"Rice", "Maize", "Wheat"} else "Apply micronutrients"
        yield_text = "Monsoon-friendly" if rainfall_value >= 800 else "Stable window"
        irrigation_text = "Water at critical growth stages" if rainfall_value < 1200 else "Monitor drainage and timing"
        pest_text = "Monitor weekly for early signs" if pest_risk == "Low to moderate" else "Increase scouting after rain"
        action_text = "Apply NPK based on crop stage" if crop_name in {"Rice", "Maize", "Wheat"} else "Adjust nutrients for soil profile"
    else:
        yield_confidence = "High" if temp_value <= 24 and rainfall_value >= 500 else "Moderate" if temp_value <= 28 and rainfall_value >= 350 else "Low"
        irrigation_plan = "Precision drip" if rainfall_value < 500 else "Smart drip"
        pest_risk = "Low" if temp_value <= 24 else "Low to moderate"
        action = "Apply basal nutrients" if crop_name in {"Wheat", "Mustard", "Barley"} else "Topdress nutrients"
        yield_text = "Cool-season fit" if season_name == "Rabi" else "Stable window"
        irrigation_text = "Water at critical early stages" if rainfall_value < 500 else "Maintain timely irrigation"
        pest_text = "Check for late blight and pest stress" if crop_name in {"Potato", "Wheat"} else "Monitor weekly for early signs"
        action_text = "Apply NPK based on crop stage" if crop_name in {"Wheat", "Rice", "Maize"} else "Adjust nutrients for soil profile"

    if "clay" in soil_lower:
        yield_confidence = "Moderate" if yield_confidence == "High" else yield_confidence
        irrigation_plan = "Controlled drainage"
        irrigation_text = "Avoid waterlogging and schedule drainage"
        pest_risk = "Moderate"
        pest_text = "Monitor humidity-linked pests after rain"
    elif "sandy" in soil_lower:
        irrigation_plan = "Frequent micro-irrigation"
        irrigation_text = "Add light irrigation during dry windows"
        action = "Use split nutrient dosing"
        action_text = "Apply split NPK to reduce leaching"

    if district_name.lower() in {"jaipur", "jodhpur", "udaipur"}:
        yield_confidence = "Moderate"
        irrigation_plan = "Moisture-saving drip"
        irrigation_text = "Prioritize water conservation and dry-spell tracking"
    elif district_name.lower() in {"chennai", "coimbatore", "bengaluru"}:
        pest_risk = "Moderate"
        pest_text = "Monitor humidity-driven disease pressure"

    return [
        {
            "label": "Yield confidence",
            "title": yield_confidence,
            "subtitle": yield_text,
            "style": "green",
        },
        {
            "label": "Irrigation plan",
            "title": irrigation_plan,
            "subtitle": irrigation_text,
            "style": "amber",
        },
        {
            "label": "Pest risk",
            "title": pest_risk,
            "subtitle": pest_text,
            "style": "blue",
        },
        {
            "label": "Action",
            "title": action,
            "subtitle": action_text,
            "style": "purple",
        },
    ]


def build_form_context(
    state: str = "Punjab",
    district: str | None = None,
    crop: str = "Rice",
    region: str | None = None,
    season: str = "Kharif",
):
    region = region or get_region_for_state(state)
    district = district or get_districts_for_state(state)[0]
    weather = get_weather_for_district(state, district)
    soil = get_soil_for_district(state, district)
    soil_options = get_soil_types_for_state(state, district)
    reference = get_reference_values(state, district, crop, season)
    historical = get_historical_yield_trend(state, district, crop, season)
    advisory = generate_agronomy_adjustments(
        state,
        district,
        soil["Soil_Type"],
        crop,
        reference["Rainfall_mm"],
        reference["Temperature_Celsius"],
        season,
    )
    feature_cards = build_feature_cards(
        state,
        district,
        crop,
        season,
        soil["Soil_Type"],
        reference["Rainfall_mm"],
        reference["Temperature_Celsius"],
    )

    # Initial baseline model prediction
    model = train_model()
    initial_yield = predict_crop_yield(
        model,
        Region=region,
        Soil_Type=soil["Soil_Type"],
        Crop=crop,
        Rainfall_mm=reference["Rainfall_mm"],
        Temperature_Celsius=reference["Temperature_Celsius"],
        Fertilizer_Used=True,
        Irrigation_Used=True,
        Weather_Condition=weather["Weather_Condition"],
        Days_to_Harvest=120,
    )
    confidence = calculate_prediction_confidence(
        model,
        Region=region,
        Soil_Type=soil["Soil_Type"],
        Crop=crop,
        Rainfall_mm=reference["Rainfall_mm"],
        Temperature_Celsius=reference["Temperature_Celsius"],
        Fertilizer_Used=True,
        Irrigation_Used=True,
        Weather_Condition=weather["Weather_Condition"],
        Days_to_Harvest=120,
    )
    economics = calculate_crop_economics(
        crop=crop,
        yield_tons_per_ha=initial_yield,
        land_area_ha=1.0,
        fertilizer_used=True,
        irrigation_used=True,
    )
    explanation = explain_prediction(
        predicted_yield=initial_yield,
        rainfall_mm=reference["Rainfall_mm"],
        temperature_celsius=reference["Temperature_Celsius"],
        fertilizer_used=True,
        irrigation_used=True,
        crop=crop,
        soil_type=soil["Soil_Type"],
        days_to_harvest=120,
    )

    return {
        "State": state,
        "District": district,
        "Region": region,
        "Crop": crop,
        "Season": season,
        "Rainfall_mm": reference["Rainfall_mm"],
        "Temperature_Celsius": reference["Temperature_Celsius"],
        "Soil_Type": soil["Soil_Type"],
        "Fertilizer_Used": "True",
        "Irrigation_Used": "True",
        "Weather_Condition": weather["Weather_Condition"],
        "Days_to_Harvest": 120,
        "crop_options": get_crop_suggestions(state),
        "district_options": get_districts_for_state(state),
        "state_options": get_states_for_region(region),
        "region_options": get_regions(),
        "soil_options": soil_options,
        "season_options": ["Kharif", "Rabi"],
        "yield_last_year": historical["last_year"],
        "yield_two_years_ago": historical["two_years_ago"],
        "yield_three_years_ago": historical["three_years_ago"],
        "nutrient_recommendation": advisory["nutrient_recommendation"],
        "seasonal_actions": advisory["seasonal_actions"],
        "feature_cards": feature_cards,
        "model_benchmark": benchmark_models(),
        "global_feature_importances": get_global_feature_importances(),
        "predicted_yield": f"{initial_yield:.2f}",
        "prediction_confidence": f"{confidence['score']:.1f}%",
        "prediction_confidence_label": confidence["label"],
        "prediction_confidence_explanation": confidence["explanation"],
        "economics": economics,
        "explanation": explanation,
        "region_to_states": REGION_TO_STATES,
        "state_to_districts": state_district_map,
        "district_climate": DISTRICT_CLIMATE,
        "state_to_region": STATE_TO_REGION,
        "state_crop_suggestions": STATE_CROP_SUGGESTIONS,
    }


@app.route("/")
def home():
    context = build_form_context()
    return render_template("index.html", **context)


@app.route("/predict", methods=["POST"])
def predict():
    model = train_model()
    region = request.form.get("Region") or "North"
    state = request.form.get("State", "Punjab")
    if state not in get_states_for_region(region):
        state = get_states_for_region(region)[0]
    district = request.form.get("District", get_districts_for_state(state)[0])
    crop = request.form.get("Crop", "Rice")
    season = request.form.get("Season", "Kharif")
    weather = get_weather_for_district(state, district)
    soil = get_soil_for_district(state, district)
    reference = get_reference_values(state, district, crop, season)
    soil_options = get_soil_types_for_state(state, district)
    historical = get_historical_yield_trend(state, district, crop, season)

    form_data = {
        "State": state,
        "District": district,
        "Region": region,
        "Season": season,
        "Soil_Type": request.form.get("Soil_Type") or soil["Soil_Type"],
        "Crop": crop,
        "Rainfall_mm": request.form.get("Rainfall_mm") or reference["Rainfall_mm"],
        "Temperature_Celsius": request.form.get("Temperature_Celsius") or reference["Temperature_Celsius"],
        "Fertilizer_Used": request.form.get("Fertilizer_Used", "True"),
        "Irrigation_Used": request.form.get("Irrigation_Used", "True"),
        "Weather_Condition": request.form.get("Weather_Condition") or weather["Weather_Condition"],
        "Days_to_Harvest": request.form.get("Days_to_Harvest", 120),
        "crop_options": get_crop_suggestions(state),
        "district_options": get_districts_for_state(state),
        "state_options": get_states_for_region(region),
        "region_options": get_regions(),
        "soil_options": soil_options,
        "season_options": ["Kharif", "Rabi"],
        "yield_last_year": historical["last_year"],
        "yield_two_years_ago": historical["two_years_ago"],
        "yield_three_years_ago": historical["three_years_ago"],
        "region_to_states": REGION_TO_STATES,
        "state_to_districts": state_district_map,
        "district_climate": DISTRICT_CLIMATE,
        "state_to_region": STATE_TO_REGION,
        "state_crop_suggestions": STATE_CROP_SUGGESTIONS,
    }

    advisory = generate_agronomy_adjustments(
        state,
        district,
        form_data["Soil_Type"],
        crop,
        float(form_data["Rainfall_mm"]),
        float(form_data["Temperature_Celsius"]),
        season,
    )
    form_data["nutrient_recommendation"] = advisory["nutrient_recommendation"]
    form_data["seasonal_actions"] = advisory["seasonal_actions"]
    form_data["feature_cards"] = build_feature_cards(
        state,
        district,
        crop,
        season,
        form_data["Soil_Type"],
        float(form_data["Rainfall_mm"]),
        float(form_data["Temperature_Celsius"]),
    )
    form_data["model_benchmark"] = benchmark_models()
    form_data["global_feature_importances"] = get_global_feature_importances()

    recommendation_text = generate_crop_recommendation(
        state=state,
        district=district,
        soil_type=form_data["Soil_Type"],
        crop=crop,
        rainfall=float(form_data["Rainfall_mm"]),
        temperature=float(form_data["Temperature_Celsius"]),
    )

    try:
        predicted_yield = predict_crop_yield(model, **form_data)
        confidence = calculate_prediction_confidence(model, **form_data)
        fert_bool = str(form_data["Fertilizer_Used"]).lower() in {"true", "yes", "1"}
        irrig_bool = str(form_data["Irrigation_Used"]).lower() in {"true", "yes", "1"}
        area_ha = float(request.form.get("Land_Area_Ha", 1.0))
        economics = calculate_crop_economics(
            crop=crop,
            yield_tons_per_ha=predicted_yield,
            land_area_ha=area_ha,
            fertilizer_used=fert_bool,
            irrigation_used=irrig_bool,
        )
        explanation = explain_prediction(
            predicted_yield=predicted_yield,
            rainfall_mm=float(form_data["Rainfall_mm"]),
            temperature_celsius=float(form_data["Temperature_Celsius"]),
            fertilizer_used=fert_bool,
            irrigation_used=irrig_bool,
            crop=crop,
            soil_type=form_data["Soil_Type"],
            days_to_harvest=int(form_data["Days_to_Harvest"]),
        )
    except (TypeError, ValueError):
        return render_template(
            "index.html",
            error="Please enter valid values for all fields.",
            recommendation_text=recommendation_text,
            **form_data,
        )

    return render_template(
        "index.html",
        predicted_yield=f"{confidence['predicted_yield']:.2f}",
        prediction_confidence=f"{confidence['score']:.1f}%",
        prediction_confidence_label=confidence["label"],
        prediction_confidence_explanation=confidence["explanation"],
        recommendation_text=recommendation_text,
        economics=economics,
        explanation=explanation,
        **form_data,
    )


# ============================================================================
# REST API Endpoints for Modern ML Deployment & Frontend Asynchronous Calls
# ============================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for real-time model inference and agricultural analytics."""
    data = request.get_json(silent=True) or request.form.to_dict()
    model = train_model()

    state = data.get("State", "Punjab")
    region = data.get("Region") or get_region_for_state(state)
    district = data.get("District") or get_districts_for_state(state)[0]
    crop = data.get("Crop", "Rice")
    season = data.get("Season", "Kharif")
    soil = get_soil_for_district(state, district)
    weather = get_weather_for_district(state, district)
    reference = get_reference_values(state, district, crop, season)

    soil_type = data.get("Soil_Type") or soil["Soil_Type"]
    rainfall = float(data.get("Rainfall_mm") or reference["Rainfall_mm"])
    temperature = float(data.get("Temperature_Celsius") or reference["Temperature_Celsius"])
    fertilizer_raw = data.get("Fertilizer_Used", "True")
    irrigation_raw = data.get("Irrigation_Used", "True")
    fertilizer_used = str(fertilizer_raw).lower() in {"true", "yes", "1"}
    irrigation_used = str(irrigation_raw).lower() in {"true", "yes", "1"}
    days_to_harvest = int(data.get("Days_to_Harvest", 120))
    weather_condition = data.get("Weather_Condition") or weather["Weather_Condition"]
    land_area_ha = float(data.get("Land_Area_Ha", 1.0))

    features = {
        "Region": region,
        "Soil_Type": soil_type,
        "Crop": crop,
        "Rainfall_mm": rainfall,
        "Temperature_Celsius": temperature,
        "Fertilizer_Used": fertilizer_used,
        "Irrigation_Used": irrigation_used,
        "Weather_Condition": weather_condition,
        "Days_to_Harvest": days_to_harvest,
    }

    try:
        predicted_yield = predict_crop_yield(model, **features)
        confidence = calculate_prediction_confidence(model, **features)
        economics = calculate_crop_economics(
            crop=crop,
            yield_tons_per_ha=predicted_yield,
            land_area_ha=land_area_ha,
            fertilizer_used=fertilizer_used,
            irrigation_used=irrigation_used,
        )
        explanation = explain_prediction(
            predicted_yield=predicted_yield,
            rainfall_mm=rainfall,
            temperature_celsius=temperature,
            fertilizer_used=fertilizer_used,
            irrigation_used=irrigation_used,
            crop=crop,
            soil_type=soil_type,
            days_to_harvest=days_to_harvest,
        )
        advisory = generate_agronomy_adjustments(
            state, district, soil_type, crop, rainfall, temperature, season
        )
        recommendation_text = generate_crop_recommendation(
            state, district, soil_type, crop, rainfall, temperature
        )
        feature_cards = build_feature_cards(
            state, district, crop, season, soil_type, rainfall, temperature
        )

        return jsonify({
            "status": "success",
            "predicted_yield": round(predicted_yield, 2),
            "confidence": confidence,
            "economics": economics,
            "explanation": explanation,
            "feature_cards": feature_cards,
            "nutrient_recommendation": advisory["nutrient_recommendation"],
            "seasonal_actions": advisory["seasonal_actions"],
            "recommendation_text": recommendation_text,
            "inputs": features,
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/api/models", methods=["GET"])
def api_models():
    """Return model comparison benchmarks across all evaluated algorithms."""
    benchmarks = benchmark_models()
    return jsonify({
        "status": "success",
        "benchmark": benchmarks,
        "total_models": len(benchmarks),
    })


@app.route("/api/feature-importance", methods=["GET"])
def api_feature_importance():
    """Return global feature importances extracted from the machine learning models."""
    importances = get_global_feature_importances()
    return jsonify({
        "status": "success",
        "feature_importances": importances,
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Service health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Crop Yield Predictor ML API",
        "version": "2.0.0",
        "model_loaded": True,
    })


@app.route("/api/states", methods=["GET"])
def api_states():
    """Return states list for a given region."""
    region = request.args.get("region", "South")
    states = get_states_for_region(region)
    return jsonify({"region": region, "states": states})


@app.route("/api/districts", methods=["GET"])
def api_districts():
    """Return districts list for a given state."""
    state = request.args.get("state", "Punjab")
    districts = get_districts_for_state(state)
    return jsonify({"state": state, "districts": districts})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
