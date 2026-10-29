import pytest
from src.economics import (
    CROP_BASE_COSTS,
    CROP_MARKET_PRICES,
    calculate_crop_economics,
)


def test_calculate_crop_economics_returns_valid_structure():
    result = calculate_crop_economics(
        crop="Rice",
        yield_tons_per_ha=6.5,
        land_area_ha=2.0,
        fertilizer_used=True,
        irrigation_used=True,
    )

    assert result["crop"] == "Rice"
    assert result["land_area_ha"] == 2.0
    assert result["yield_tons_per_ha"] == 6.5
    assert result["total_yield_tons"] == 13.0
    assert result["price_per_ton_inr"] == CROP_MARKET_PRICES["Rice"]
    assert result["gross_revenue_inr"] == 13.0 * CROP_MARKET_PRICES["Rice"]
    assert result["total_cost_inr"] > 0
    assert result["net_profit_inr"] == result["gross_revenue_inr"] - result["total_cost_inr"]
    assert result["is_profitable"] is True
    assert result["roi_percent"] > 0


def test_economics_zero_inputs_reduces_costs():
    with_inputs = calculate_crop_economics("Wheat", yield_tons_per_ha=4.0, fertilizer_used=True, irrigation_used=True)
    without_inputs = calculate_crop_economics("Wheat", yield_tons_per_ha=4.0, fertilizer_used=False, irrigation_used=False)

    assert with_inputs["total_cost_inr"] > without_inputs["total_cost_inr"]


def test_economics_handles_unsupported_crop_gracefully():
    result = calculate_crop_economics("UnknownCrop", yield_tons_per_ha=3.0)
    assert result["gross_revenue_inr"] > 0
    assert result["total_cost_inr"] > 0
