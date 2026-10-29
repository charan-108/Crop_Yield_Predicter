from __future__ import annotations

import concurrent.futures
import os
from typing import Any

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


def _call_gemini_with_timeout(model, prompt: str, timeout: float = 5.0):
    """Run model.generate_content in a thread with a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.generate_content, prompt)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Gemini API call timed out")


def _fallback_recommendation(state: str, district: str, soil_type: str, crop: str, rainfall: float, temperature: float) -> str:
    return (
        f"For {district}, {state}, the selected crop {crop} is suited to the current soil profile "
        f"({soil_type}) and climate conditions. With rainfall around {rainfall:.0f} mm and temperature "
        f"around {temperature:.1f}°C, focus on balanced irrigation, nutrient timing, and crop monitoring "
        f"to maintain healthy growth and yield."
    )


def _parse_gemini_list(text: str) -> list[str]:
    if not text:
        return []
    items: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = cleaned.lstrip("-•*0123456789. ")
        if cleaned:
            items.append(cleaned)
    return items[:3]


def generate_agronomy_adjustments(
    state: str,
    district: str,
    soil_type: str,
    crop: str,
    rainfall: float,
    temperature: float,
    season: str,
) -> dict[str, list[str]]:
    """Return crop-, soil-, district-, and season-specific guidance.

    If Gemini is available with an API key, it generates the advice from the field context.
    Otherwise a deterministic local set of recommendations is used.
    """
    api_key = None if (os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING")) else os.getenv("GEMINI_API_KEY")
    if api_key and genai is not None:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are an agricultural advisor. Return only two sections in plain text. "
                "Section 1: 'Nutrient Recommendation:' followed by exactly 3 bullet points. "
                "Section 2: 'Seasonal Action Checklist:' followed by exactly 3 bullet points. "
                "Keep each point practical and specific to the selected crop, soil, district, and season. "
                f"State: {state}. District: {district}. Soil: {soil_type}. Crop: {crop}. "
                f"Rainfall: {rainfall:.1f} mm. Temperature: {temperature:.1f} C. Season: {season}."
            )
            response = _call_gemini_with_timeout(model, prompt, timeout=5.0)
            text = getattr(response, "text", None)
            if text and text.strip():
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                nutrient_lines: list[str] = []
                seasonal_lines: list[str] = []
                current_section: str | None = None
                for line in lines:
                    upper = line.upper()
                    if "NUTRIENT RECOMMENDATION" in upper:
                        current_section = "nutrient"
                        continue
                    if "SEASONAL ACTION CHECKLIST" in upper:
                        current_section = "seasonal"
                        continue
                    if line.startswith("-") or line.startswith("•") or any(line.startswith(prefix) for prefix in ("1.", "2.", "3.")):
                        cleaned = line.lstrip("-•*0123456789. ")
                        if current_section == "nutrient":
                            nutrient_lines.append(cleaned)
                        elif current_section == "seasonal":
                            seasonal_lines.append(cleaned)
                if nutrient_lines or seasonal_lines:
                    return {
                        "nutrient_recommendation": nutrient_lines[:3] or _parse_gemini_list("- Apply balanced nutrition.")[:3],
                        "seasonal_actions": seasonal_lines[:3] or _parse_gemini_list("- Follow seasonal field scheduling.")[:3],
                    }
        except Exception:
            pass

    def _deduplicate(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            result.append(normalized)
        return result

    def _soil_adjustments(crop_name: str, soil_name: str) -> list[str]:
        soil_lower = (soil_name or "").lower()
        if "clay" in soil_lower:
            return [
                f"Improve drainage and avoid waterlogging to protect {crop_name} roots in heavy clay soil.",
                "Apply organic matter to loosen the profile and improve nutrient uptake.",
                "Use gypsum or split amendments where soil structure is compacted during wet spells.",
            ]
        if "sandy" in soil_lower:
            return [
                f"Increase split nitrogen applications for {crop_name} to reduce leaching losses in sandy soil.",
                "Use mulch and frequent monitoring to reduce moisture stress during dry periods.",
                "Add organic matter to improve nutrient retention and root stability.",
            ]
        if "alluvial" in soil_lower or "loam" in soil_lower:
            return [
                f"Use balanced nutrient splits to keep {crop_name} performance high in fertile alluvial or loamy soil.",
                "Maintain consistent irrigation timing to preserve nutrient efficiency in this soil.",
                "Use soil testing to confirm micronutrient sufficiency before the peak vegetative stage.",
            ]
        if "laterite" in soil_lower or "red" in soil_lower:
            return [
                f"Apply phosphorus carefully to improve root development and establishment for {crop_name} in red or lateritic soil.",
                "Protect soil organic matter to support structure and soil moisture retention.",
                "Apply micronutrients judiciously to correct acidity-driven nutrient limits.",
            ]
        return [
            f"Adjust nutrient timing to match the current soil texture for {crop_name} growth and yield.",
            "Use soil testing to fine-tune the fertilizer plan as the season progresses.",
            "Maintain a balanced micronutrient program for stable crop performance.",
        ]

    def _season_nutrient_adjustments(season_name: str) -> list[str]:
        if season_name == "Kharif":
            return [
                "Increase nitrogen efficiency management to account for heavier rain and possible leaching.",
                "Plan split applications around monsoon breaks and peak vegetative growth windows.",
            ]
        if season_name == "Rabi":
            return [
                "Use cooler-season nutrient timing with stronger focus on root establishment and grain fill.",
                "Keep fertilizer application aligned with residual moisture and low-temperature stress.",
            ]
        return [
            "Adjust nutrient timing to the current field conditions and crop stage.",
            "Monitor deficiencies regularly as the season changes.",
        ]

    crop_name = (crop or "Rice").strip().title()
    season_name = (season or "Kharif").strip().title()
    district_name = (district or "Amritsar").strip().title()
    soil_name = (soil_type or "Loam").strip()

    nutrient_map = {
        "Rice": [
            "Apply phosphorus and zinc at transplanting for healthy root establishment.",
            "Split nitrogen into 3 to 4 applications to match vegetative growth and panicle development.",
            "Use potassium and micronutrients to reduce lodging and improve grain filling under wet conditions.",
        ],
        "Wheat": [
            "Apply phosphorus and potassium at sowing to support early root growth.",
            "Topdress nitrogen in splits during crown root initiation and flowering.",
            "Use sulfur and zinc where soil tests show deficiency for better grain quality.",
        ],
        "Maize": [
            "Apply basal NPK before sowing for strong stand establishment.",
            "Topdress nitrogen at knee-high and tasseling to improve ear development.",
            "Maintain potassium and micronutrient supply to support cob filling and stress tolerance.",
        ],
        "Cotton": [
            "Base the fertilizer schedule on a balanced NPK plan at planting and early square formation.",
            "Use potassium and boron to improve boll retention and fiber strength.",
            "Avoid excess nitrogen late in the crop cycle to reduce pest pressure and fruit shedding.",
        ],
        "Soybean": [
            "Apply phosphorus and sulfur for nodulation and pod set.",
            "Use a light nitrogen starter only if soil fertility is low.",
            "Maintain potassium and moisture during flowering to reduce stress-related yield losses.",
        ],
        "Sugarcane": [
            "Apply basal NPK and organic manure at planting for vigorous cane establishment.",
            "Split nitrogen across the grand growth phase to maximize cane elongation.",
            "Provide potassium and micronutrients to support sucrose build-up and stress resistance.",
        ],
        "Groundnut": [
            "Use phosphorus and calcium at sowing to support pod formation and shelling quality.",
            "Apply sulfur to improve kernel filling and oil development.",
            "Keep soil moisture steady around pegging and pod filling to avoid pod abortion.",
        ],
        "Mustard": [
            "Apply a basal dose of phosphorus and sulfur before sowing.",
            "Use split nitrogen around early vegetative growth and flowering.",
            "Maintain balanced potassium to support seed set and oil content in cool conditions.",
        ],
        "Barley": [
            "Apply basal phosphorus and nitrogen early for a strong tillering stage.",
            "Use split nitrogen to improve grain fill and reduce stress during spike emergence.",
            "Supply potassium and micronutrients in balanced form for stress tolerance.",
        ],
        "Pearl Millet": [
            "Apply phosphorus and nitrogen at sowing for vigorous early growth.",
            "Use split nitrogen during tillering and ear emergence to sustain productivity.",
            "Maintain moisture and potash to improve grain set in dry conditions.",
        ],
        "Potato": [
            "Apply phosphorus and potassium at planting for tuber initiation and strong root growth.",
            "Use split nitrogen during vegetative growth and tuber bulking.",
            "Ensure calcium and micronutrient balance to reduce defects and improve size.",
        ],
    }

    seasonal_map = {
        "Kharif": [
            "Prepare field drainage and bunding before monsoon onset to prevent waterlogging.",
            "Use irrigation only during dry spells and maintain consistent moisture during critical growth stages.",
            "Monitor pest and disease pressure after rainfall events and control weeds early.",
        ],
        "Rabi": [
            "Prepare a fine seedbed and conserve residual moisture for timely sowing.",
            "Apply light irrigation at critical stages and protect plants from frost or cold stress.",
            "Keep crop monitoring tight during flowering and grain filling to avoid nutrient stress.",
        ],
    }

    district_notes = {
        "Amritsar": [
            "Plan water and nutrient management carefully in the alluvial plains during monsoon swings.",
            "Use field drainage and pest scouting for repeated cereal rotations in this district.",
        ],
        "Ludhiana": [
            "Maintain efficient irrigation scheduling in the intensively cropped alluvial belt.",
            "Use nutrient balancing to protect productivity in high-input wheat-rice systems.",
        ],
        "Patiala": [
            "Monitor drainage and soil moisture carefully in the irrigated belt.",
            "Avoid excess water near the reproductive stages for cereals and pulses.",
        ],
        "Jaipur": [
            "Prioritize moisture conservation and mulch use to reduce heat stress.",
            "Use targeted irrigation and nutrient application during dry spells.",
        ],
        "Jodhpur": [
            "Favor moisture-retaining practices and low-volume irrigation under dry conditions.",
            "Reduce nutrient loss by applying tailored doses during active growth windows.",
        ],
        "Chennai": [
            "Watch humidity-related disease pressure and maintain proper drainage after rain.",
            "Adjust irrigation to reduce water stress while preventing disease-friendly moisture.",
        ],
        "Coimbatore": [
            "Keep irrigation and disease protection aligned with warm, variable weather.",
            "Use soil condition checks to time nutrient application more precisely.",
        ],
        "Bengaluru": [
            "Avoid waterlogging and monitor pest pressure during wet spells.",
            "Fine-tune nutrient timings to match soil moisture and crop stage.",
        ],
        "Mumbai": [
            "Be prepared for heavy rainfall and poor drainage in monsoon months.",
            "Keep field access and pest scouting consistent after storm events.",
        ],
        "Kolkata": [
            "Keep field drainage active during prolonged wet periods.",
            "Use staged nutrient application to support growth during humid months.",
        ],
        "Dehradun": [
            "Protect the crop from sudden rainfall spikes and manage runoff in the foothill belt.",
            "Align nutrient application with variable hill-to-plain rainfall patterns.",
        ],
        "Nagpur": [
            "Manage soil moisture carefully during the monsoon and maintain drainage around the crop roots.",
            "Use staged fertilizer application to support high-yield cereal and cotton systems.",
        ],
    }

    district_specific = district_notes.get(district_name, [
        f"Keep operations aligned with the local rainfall and field conditions in {district_name}.",
        "Adjust nutrient and irrigation timing based on actual soil moisture and crop growth stage.",
    ])
    seasonal_actions = seasonal_map.get(season_name, [
        "Adjust irrigation to field moisture and rainfall patterns.",
        "Monitor weeds, pests, and disease pressure regularly.",
        "Review nutrient timing with the current seasonal conditions.",
    ])

    if "clay" in soil_name.lower() and season_name == "Kharif":
        seasonal_actions = [
            "Open field drainage channels early and avoid stagnant water in clay soils.",
            *seasonal_actions[1:],
        ]
    elif "sandy" in soil_name.lower() and season_name == "Rabi":
        seasonal_actions = [
            "Use frequent moisture checks and light irrigation in sandy soil to protect the crop during dry spells.",
            *seasonal_actions[1:],
        ]

    nutrient_recommendation = nutrient_map.get(crop_name, [
        "Apply a balanced NPK program based on soil tests and crop stage.",
        "Use split nitrogen applications to reduce loss and improve efficiency.",
        "Monitor the field for deficiency symptoms and adjust micronutrients accordingly.",
    ])
    ordered_nutrients = _soil_adjustments(crop_name, soil_name) + _season_nutrient_adjustments(season_name) + nutrient_recommendation
    nutrient_recommendation = _deduplicate(ordered_nutrients)[:3]

    soil_specific_seasonal = []
    if "clay" in soil_name.lower():
        soil_specific_seasonal.extend([
            "Open field drainage channels early and avoid stagnant water in clay soils.",
            "Improve soil structure with organic matter and controlled drainage planning.",
        ])
    elif "sandy" in soil_name.lower():
        soil_specific_seasonal.extend([
            "Use frequent moisture checks and light irrigation in sandy soil to protect the crop during dry spells.",
            "Reduce nutrient leaching with split applications and mulching.",
        ])
    seasonal_actions = _deduplicate(seasonal_actions + district_specific + soil_specific_seasonal)[:3]

    return {
        "nutrient_recommendation": nutrient_recommendation,
        "seasonal_actions": seasonal_actions,
    }


def build_agronomy_context(
    state: str,
    district: str,
    soil_type: str,
    crop: str,
    rainfall: float,
    temperature: float,
    season: str,
    weather_condition: str | None = None,
    fertilizer_used: bool | None = None,
    irrigation_used: bool | None = None,
    days_to_harvest: int | None = None,
) -> dict[str, Any]:
    """Return a structured agricultural summary covering all fields that influence recommendations."""
    crop_name = (crop or "Rice").strip().title()
    soil_name = (soil_type or "Loam").strip()
    season_name = (season or "Kharif").strip().title()
    weather_name = (weather_condition or "Sunny").strip().title()

    nutrient_need = {
        "crop": crop_name,
        "soil": soil_name,
        "priority": "Nitrogen + phosphorus balance",
        "timing": "Split applications during active growth stages",
        "notes": [
            f"For {crop_name} in {soil_name} soil, manage nutrient release to match rainfall and root activity.",
            f"Adjust fertilizer timing for {season_name} field conditions in {district}, {state}.",
        ],
    }

    risk_summary = {
        "water_stress": "Moderate" if rainfall < 600 else "Low",
        "heat_stress": "High" if temperature > 30 else "Moderate",
        "pest_pressure": "Moderate" if weather_name.lower() in {"rainy", "cloudy"} else "Low",
        "drainage_risk": "High" if "clay" in soil_name.lower() else "Moderate" if "sandy" in soil_name.lower() else "Low",
    }

    agronomy_profile = {
        "state": (state or "Punjab").strip().title(),
        "district": (district or "Amritsar").strip().title(),
        "soil_type": soil_name,
        "crop": crop_name,
        "season": season_name,
        "rainfall_mm": round(float(rainfall or 0), 2),
        "temperature_c": round(float(temperature or 0), 2),
        "weather_condition": weather_name,
        "fertilizer_used": bool(fertilizer_used),
        "irrigation_used": bool(irrigation_used),
        "days_to_harvest": int(days_to_harvest or 120),
        "nutrient_need": nutrient_need,
        "risk_summary": risk_summary,
        "local_context": {
            "district_note": f"{district}, {state} requires field management adapted to local rainfall and soil behavior.",
            "season_note": f"{season_name} management should prioritize drainage, moisture control, and crop-stage nutrition.",
        },
        "recommendation_summary": {
            "yield_confidence": "High" if rainfall >= 700 and temperature <= 32 else "Moderate",
            "irrigation_plan": "Smart drip" if rainfall >= 600 else "Targeted drip",
            "pest_risk": "Low to moderate" if weather_name.lower() in {"sunny", "cloudy"} else "Moderate",
            "action": "Topdress nutrients" if crop_name in {"Rice", "Maize", "Wheat"} else "Apply micronutrients",
        },
    }

    return agronomy_profile


def generate_crop_recommendation(
    state: str,
    district: str,
    soil_type: str,
    crop: str,
    rainfall: float,
    temperature: float,
) -> str:
    """Generate an agronomic recommendation.

    If a Gemini API key is configured and the package is available, the app uses Gemini for a richer
    recommendation. Otherwise it falls back to a deterministic local recommendation.
    """
    api_key = None if (os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING")) else os.getenv("GEMINI_API_KEY")
    if api_key and genai is not None:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are an agricultural advisor. Based on this district profile, give a short, practical "
                "recommendation for the chosen crop. Keep the guidance concise and actionable. "
                f"State: {state}. District: {district}. Soil: {soil_type}. Crop: {crop}. "
                f"Rainfall: {rainfall:.1f} mm. Temperature: {temperature:.1f} C."
            )
            response = _call_gemini_with_timeout(model, prompt, timeout=5.0)
            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip()
        except Exception:
            pass
    return _fallback_recommendation(state, district, soil_type, crop, rainfall, temperature)
