from __future__ import annotations

from typing import Any

# Standard Minimum Support Price (MSP) and average market price in INR per ton (1 ton = 10 quintals)
# Reference values based on recent agricultural market data in India
CROP_MARKET_PRICES = {
    "Rice": 23000.0,       # ~2,300 INR/quintal
    "Wheat": 24250.0,      # ~2,425 INR/quintal
    "Maize": 22250.0,      # ~2,225 INR/quintal
    "Cotton": 75210.0,     # ~7,521 INR/quintal (raw cotton)
    "Barley": 18500.0,     # ~1,850 INR/quintal
    "Soybean": 48920.0,    # ~4,892 INR/quintal
}

# Estimated baseline input production cost in INR per hectare (seed, labor, land prep, machinery)
CROP_BASE_COSTS = {
    "Rice": 38000.0,
    "Wheat": 32000.0,
    "Maize": 28000.0,
    "Cotton": 45000.0,
    "Barley": 24000.0,
    "Soybean": 30000.0,
}

# Variable costs per hectare
FERTILIZER_COST_PER_HA = 6500.0
IRRIGATION_COST_PER_HA = 5000.0


def calculate_crop_economics(
    crop: str,
    yield_tons_per_ha: float,
    land_area_ha: float = 1.0,
    fertilizer_used: bool = True,
    irrigation_used: bool = True,
) -> dict[str, Any]:
    """Calculate projected gross revenue, input expenses, and net profit for a given crop yield."""
    crop_name = (crop or "Rice").strip().title()
    area = max(0.1, float(land_area_ha or 1.0))
    yield_per_ha = max(0.0, float(yield_tons_per_ha or 0.0))
    total_yield_tons = round(yield_per_ha * area, 2)

    price_per_ton = CROP_MARKET_PRICES.get(crop_name, 25000.0)
    base_cost_per_ha = CROP_BASE_COSTS.get(crop_name, 32000.0)

    fert_cost = FERTILIZER_COST_PER_HA if fertilizer_used else 0.0
    irrig_cost = IRRIGATION_COST_PER_HA if irrigation_used else 0.0
    cost_per_ha = base_cost_per_ha + fert_cost + irrig_cost
    total_cost = round(cost_per_ha * area, 2)

    gross_revenue = round(total_yield_tons * price_per_ton, 2)
    net_profit = round(gross_revenue - total_cost, 2)
    roi_percent = round((net_profit / total_cost * 100.0) if total_cost > 0 else 0.0, 1)

    return {
        "crop": crop_name,
        "land_area_ha": area,
        "yield_tons_per_ha": round(yield_per_ha, 2),
        "total_yield_tons": total_yield_tons,
        "price_per_ton_inr": price_per_ton,
        "price_per_quintal_inr": round(price_per_ton / 10.0, 2),
        "total_cost_inr": total_cost,
        "cost_per_ha_inr": round(cost_per_ha, 2),
        "gross_revenue_inr": gross_revenue,
        "net_profit_inr": net_profit,
        "roi_percent": roi_percent,
        "is_profitable": net_profit > 0,
    }
