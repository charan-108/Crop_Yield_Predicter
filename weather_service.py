from __future__ import annotations

import os
from typing import Any

import requests

from src.crop_yield_predictor import DISTRICT_CLIMATE, get_districts_for_state


DEFAULT_WEATHER = {"Rainfall_mm": 550.0, "Temperature_Celsius": 27.5, "Weather_Condition": "Sunny"}


def _fallback_weather(state: str, district: str | None = None) -> dict[str, Any]:
    state_name = (state or "Punjab").strip().title()
    district_name = (district or get_districts_for_state(state_name)[0]).strip().title()
    climate = DISTRICT_CLIMATE.get(district_name, DEFAULT_WEATHER)
    return {
        "State": state_name,
        "District": district_name,
        "Rainfall_mm": float(climate.get("Rainfall_mm", DEFAULT_WEATHER["Rainfall_mm"])),
        "Temperature_Celsius": float(climate.get("Temperature_Celsius", DEFAULT_WEATHER["Temperature_Celsius"])),
        "Weather_Condition": "Rainy" if float(climate.get("Rainfall_mm", 550.0)) > 1200 else "Sunny" if float(climate.get("Temperature_Celsius", 27.5)) > 29 else "Cloudy",
    }


def _fetch_live_weather(state: str, district: str | None = None) -> dict[str, Any] | None:
    if os.getenv('PYTEST_CURRENT_TEST') or os.getenv('TESTING') or os.getenv('SKIP_LIVE_WEATHER'):
        return None
    district_name = (district or get_districts_for_state(state or "Punjab")[0]).strip()
    api_key = os.getenv("WEATHER_API_KEY")

    try:
        if api_key:
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            weather_response = requests.get(
                weather_url,
                params={"q": district_name, "appid": api_key, "units": "metric"},
                timeout=2.0,
            )
            if weather_response.status_code == 200:
                payload = weather_response.json()
                main = payload.get("main") or {}
                weather = (payload.get("weather") or [{}])[0]
                return {
                    "State": (state or "Punjab").strip().title(),
                    "District": district_name.title(),
                    "Rainfall_mm": float(payload.get("rain", {}).get("1h", 0.0) or 0.0),
                    "Temperature_Celsius": float(main.get("temp", 27.5)),
                    "Weather_Condition": (weather.get("main") or "Sunny").title(),
                }

        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_response = requests.get(
            geo_url,
            params={"name": district_name, "count": 1, "language": "en", "format": "json"},
            timeout=2.0,
        )
        geo_response.raise_for_status()
        results = geo_response.json().get("results") or []
        if not results:
            return None
        location = results[0]
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        if latitude is None or longitude is None:
            return None
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,precipitation",
                "hourly": "temperature_2m,precipitation",
                "timezone": "auto",
            },
            timeout=2.0,
        )
        weather_response.raise_for_status()
        payload = weather_response.json()
        current = payload.get("current") or {}
        return {
            "State": (state or "Punjab").strip().title(),
            "District": district_name.title(),
            "Rainfall_mm": float(current.get("precipitation", 0.0)),
            "Temperature_Celsius": float(current.get("temperature_2m", 27.5)),
            "Weather_Condition": "Rainy" if float(current.get("precipitation", 0.0)) > 5 else "Sunny",
        }
    except Exception:
        return None


def get_weather_for_district(state: str, district: str | None = None) -> dict[str, Any]:
    """Return rainfall, temperature, and a summary condition for a district.

    Live weather data is attempted when available, but the app falls back to district metadata
    so the app remains functional even without internet access or an API key.
    """
    live_weather = _fetch_live_weather(state, district)
    if live_weather:
        return live_weather
    return _fallback_weather(state, district)
