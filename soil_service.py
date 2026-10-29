from __future__ import annotations

from typing import Any

from src.crop_yield_predictor import DISTRICT_SOILS, SOIL_TYPES, get_soil_types_for_state


def get_soil_for_district(state: str, district: str | None = None) -> dict[str, Any]:
    """Return a district-aware soil profile, with a deterministic fallback.

    This is designed to work offline while still allowing a future API integration.
    """
    state_name = (state or "Punjab").strip().title()
    district_name = (district or "").strip().title()
    soils = DISTRICT_SOILS.get(district_name, [])
    soil_type = (soils[0] if soils else get_soil_types_for_state(state_name, district_name)[0])
    soil_details = {
        "Loam": "Well-drained soil with balanced sand, silt, and clay; good for diverse crops.",
        "Sandy Loam": "Light, well-drained soil that warms quickly and suits cereals and pulses.",
        "Clay Loam": "Moisture-retentive soil with good nutrient holding capacity for cereals and oilseeds.",
        "Silt Loam": "Fertile, friable soil suited to high-yield crop growth.",
        "Sandy": "Light sandy soil with quick drainage and lower water retention.",
        "Clay": "Heavy, moisture-retentive soil that needs careful irrigation management.",
        "Alluvial Soil": "Highly fertile river-deposited soil that is ideal for intensive agriculture.",
        "Black Soil": "Clay-rich dark soil with high fertility and water-holding capacity.",
        "Red Soil": "Well-drained, iron-rich soil typically suited to millets, pulses, and cotton.",
        "Laterite Soil": "Iron-rich, porous soil common in humid regions requiring nutrient management.",
    }
    return {
        "State": state_name,
        "District": district_name or "Main District",
        "Soil_Type": soil_type,
        "Soil_Description": soil_details.get(soil_type, "A fertile agricultural soil suitable for crop growth."),
        "Available_Soil_Types": soils or get_soil_types_for_state(state_name, district_name) or SOIL_TYPES,
    }
