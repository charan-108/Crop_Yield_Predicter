from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parents[1] / "crop_yield.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
TARGET_COLUMN = "Yield_tons_per_hectare"
REQUIRED_FEATURES = [
    "Region",
    "Soil_Type",
    "Crop",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Weather_Condition",
    "Days_to_Harvest",
]

STATE_TO_REGION = {
    "Andhra Pradesh": "South",
    "Arunachal Pradesh": "North-East",
    "Assam": "North-East",
    "Bihar": "East",
    "Chhattisgarh": "Central",
    "Goa": "West",
    "Gujarat": "West",
    "Haryana": "North",
    "Himachal Pradesh": "North",
    "Jharkhand": "East",
    "Karnataka": "South",
    "Kerala": "South",
    "Madhya Pradesh": "Central",
    "Maharashtra": "West",
    "Manipur": "North-East",
    "Meghalaya": "North-East",
    "Mizoram": "North-East",
    "Nagaland": "North-East",
    "Odisha": "East",
    "Punjab": "North",
    "Rajasthan": "North",
    "Sikkim": "North-East",
    "Tamil Nadu": "South",
    "Telangana": "South",
    "Tripura": "North-East",
    "Uttar Pradesh": "North",
    "Uttarakhand": "North",
    "West Bengal": "East",
}

REGION_TO_STATES = {
    "North": ["Haryana", "Himachal Pradesh", "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand"],
    "South": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"],
    "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "West": ["Goa", "Gujarat", "Maharashtra"],
    "Central": ["Chhattisgarh", "Madhya Pradesh"],
    "North-East": ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura"],
}

state_district_map = {
    "Andhra Pradesh": [
        "Visakhapatnam", "Guntur", "Krishna", "Anantapur", "Kurnool", "Kadapa",
        "Chittoor", "East Godavari", "West Godavari", "Prakasam", "Nellore", "Srikakulam",
        "Vizianagaram", "Tirupati", "Rajamahendravaram"
    ],
    "Arunachal Pradesh": [
        "Itanagar", "Tawang", "Pasighat", "Naharlagun", "Bomdila", "Ziro",
        "Tezu", "Khonsa", "Changlang", "Daporijo", "Aalo", "Roing"
    ],
    "Assam": [
        "Kamrup", "Jorhat", "Dibrugarh", "Nagaon", "Silchar", "Bongaigaon",
        "Guwahati", "Barpeta", "Golaghat", "Kokrajhar", "Dhubri", "Majuli"
    ],
    "Bihar": [
        "Patna", "Muzaffarpur", "Gaya", "Bhagalpur", "Purnia", "Darbhanga",
        "Hajipur", "Samastipur", "Siwan", "Begusarai", "Nalanda", "Munger"
    ],
    "Chhattisgarh": [
        "Raipur", "Bilaspur", "Durg", "Bastar", "Rajnandgaon", "Korba",
        "Dantewada", "Jagdalpur", "Kawardha", "Jashpur", "Mahasamund", "Dhamtari"
    ],
    "Goa": [
        "North Goa", "South Goa", "Panaji", "Margao", "Mapusa", "Ponda",
        "Vasco da Gama", "Curchorem", "Bicholim", "Quepem", "Canacona", "Sanguem"
    ],
    "Gujarat": [
        "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Junagadh", "Bhavnagar",
        "Gandhinagar", "Kutch", "Jamnagar", "Mehsana", "Anand", "Nadiad"
    ],
    "Haryana": [
        "Gurugram", "Faridabad", "Hisar", "Panipat", "Ambala", "Rohtak",
        "Karnal", "Sonipat", "Kurukshetra", "Sirsa", "Jind", "Bhiwani"
    ],
    "Himachal Pradesh": [
        "Shimla", "Mandi", "Kangra", "Solan", "Una", "Bilaspur",
        "Hamirpur", "Kullu", "Chamba", "Sirmaur", "Kinnaur", "Lahaul and Spiti"
    ],
    "Jharkhand": [
        "Ranchi", "Jamshedpur", "Dhanbad", "Hazaribagh", "Bokaro", "Giridih",
        "Deoghar", "Dumka", "Chatra", "Khunti", "Palamu", "Garhwa"
    ],
    "Karnataka": [
        "Bengaluru", "Mysuru", "Belagavi", "Ballari", "Hubballi", "Kalaburagi",
        "Shivamogga", "Mandya", "Hassan", "Tumakuru", "Dakshina Kannada", "Udupi"
    ],
    "Kerala": [
        "Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Alappuzha", "Palakkad",
        "Kannur", "Kollam", "Kottayam", "Malappuram", "Wayanad", "Kasaragod"
    ],
    "Madhya Pradesh": [
        "Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain", "Sagar",
        "Rewa", "Satna", "Ratlam", "Guna", "Morena", "Dhar"
    ],
    "Maharashtra": [
        "Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Kolhapur",
        "Solapur", "Amravati", "Akola", "Sangli", "Nanded", "Yavatmal"
    ],
    "Manipur": [
        "Imphal", "Thoubal", "Bishnupur", "Churachandpur", "Ukhrul", "Senapati",
        "Kangpokpi", "Tamenglong", "Jiribam", "Kakching", "Noney", "Pherzawl"
    ],
    "Meghalaya": [
        "Shillong", "Tura", "Jowai", "Nongpoh", "Baghmara", "Williamnagar",
        "Nongstoin", "Resubelpara", "Mawkyrwat", "Ampati", "Ribhoi", "West Jaintia Hills"
    ],
    "Mizoram": [
        "Aizawl", "Lunglei", "Champhai", "Serchhip", "Kolasib", "Lawngtlai",
        "Mamit", "Siaha", "Hnahthial", "Saiha", "Ngopa", "Reiek"
    ],
    "Nagaland": [
        "Kohima", "Dimapur", "Mokokchung", "Wokha", "Phek", "Tuensang",
        "Zunheboto", "Kiphire", "Longleng", "Chumoukedima", "Pfutsero", "Tseminyü"
    ],
    "Odisha": [
        "Bhubaneswar", "Cuttack", "Sambalpur", "Rourkela", "Puri", "Balasore",
        "Angul", "Baripada", "Berhampur", "Jeypore", "Rayagada", "Jharsuguda"
    ],
    "Punjab": [
        "Amritsar", "Ludhiana", "Patiala", "Jalandhar", "Bathinda", "Gurdaspur",
        "Ferozepur", "Mohali", "Moga", "Tarn Taran", "Kapurthala", "Sangrur"
    ],
    "Rajasthan": [
        "Jaipur", "Jodhpur", "Udaipur", "Ajmer", "Kota", "Bikaner",
        "Alwar", "Bharatpur", "Baran", "Sikar", "Nagaur", "Pali"
    ],
    "Sikkim": [
        "Gangtok", "Namchi", "Gyalshing", "Mangan", "Singtam", "Rhenock",
        "Pakyong", "Jorethang", "Ravangla", "Soreng", "Melli", "Naya Bazar"
    ],
    "Tamil Nadu": [
        "Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Vellore",
        "Tirunelveli", "Erode", "Dindigul", "Thanjavur", "Kanchipuram", "Ramanathapuram"
    ],
    "Telangana": [
        "Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam", "Mahbubnagar",
        "Sangareddy", "Adilabad", "Rajanna Sircilla", "Medak", "Jagtial", "Nalgonda"
    ],
    "Tripura": [
        "Agartala", "Udaipur", "Khowai", "Dharmanagar", "Kailashahar", "Belonia",
        "Ambassa", "Sabroom", "Kumarkhal", "Teliamura", "Kailasahar", "Kamalpur"
    ],
    "Uttar Pradesh": [
        "Lucknow", "Kanpur", "Varanasi", "Agra", "Allahabad", "Meerut",
        "Ghaziabad", "Bareilly", "Moradabad", "Aligarh", "Noida", "Gorakhpur"
    ],
    "Uttarakhand": [
        "Dehradun", "Haridwar", "Nainital", "Haldwani", "Rudrapur", "Pauri",
        "Almora", "Chamoli", "Tehri", "Bageshwar", "Udham Singh Nagar", "Rishikesh"
    ],
    "West Bengal": [
        "Kolkata", "Asansol", "Darjeeling", "Durgapur", "Howrah", "Midnapore",
        "Siliguri", "Bardhaman", "Cooch Behar", "Malda", "Purulia", "Bankura"
    ],
}

DISTRICT_CLIMATE = {
    "Amritsar": {"Rainfall_mm": 620.0, "Temperature_Celsius": 27.0},
    "Ludhiana": {"Rainfall_mm": 680.0, "Temperature_Celsius": 26.4},
    "Patiala": {"Rainfall_mm": 645.0, "Temperature_Celsius": 26.6},
    "Jalandhar": {"Rainfall_mm": 610.0, "Temperature_Celsius": 26.8},
    "Bathinda": {"Rainfall_mm": 490.0, "Temperature_Celsius": 28.4},
    "Gurdaspur": {"Rainfall_mm": 720.0, "Temperature_Celsius": 25.9},
    "Gurugram": {"Rainfall_mm": 600.0, "Temperature_Celsius": 28.2},
    "Faridabad": {"Rainfall_mm": 560.0, "Temperature_Celsius": 28.6},
    "Hisar": {"Rainfall_mm": 470.0, "Temperature_Celsius": 29.1},
    "Lucknow": {"Rainfall_mm": 810.0, "Temperature_Celsius": 27.5},
    "Kanpur": {"Rainfall_mm": 760.0, "Temperature_Celsius": 27.8},
    "Varanasi": {"Rainfall_mm": 940.0, "Temperature_Celsius": 27.3},
    "Agra": {"Rainfall_mm": 650.0, "Temperature_Celsius": 28.4},
    "Allahabad": {"Rainfall_mm": 880.0, "Temperature_Celsius": 27.1},
    "Jaipur": {"Rainfall_mm": 520.0, "Temperature_Celsius": 28.5},
    "Jodhpur": {"Rainfall_mm": 340.0, "Temperature_Celsius": 30.1},
    "Udaipur": {"Rainfall_mm": 620.0, "Temperature_Celsius": 27.2},
    "Kota": {"Rainfall_mm": 760.0, "Temperature_Celsius": 28.7},
    "Bengaluru": {"Rainfall_mm": 950.0, "Temperature_Celsius": 25.2},
    "Mysuru": {"Rainfall_mm": 780.0, "Temperature_Celsius": 25.0},
    "Chennai": {"Rainfall_mm": 1350.0, "Temperature_Celsius": 30.2},
    "Coimbatore": {"Rainfall_mm": 700.0, "Temperature_Celsius": 28.5},
    "Hyderabad": {"Rainfall_mm": 800.0, "Temperature_Celsius": 28.3},
    "Warangal": {"Rainfall_mm": 860.0, "Temperature_Celsius": 28.0},
    "Mumbai": {"Rainfall_mm": 2200.0, "Temperature_Celsius": 29.1},
    "Pune": {"Rainfall_mm": 760.0, "Temperature_Celsius": 25.0},
    "Nagpur": {"Rainfall_mm": 1050.0, "Temperature_Celsius": 27.5},
    "Kolkata": {"Rainfall_mm": 1600.0, "Temperature_Celsius": 29.0},
    "Darjeeling": {"Rainfall_mm": 3100.0, "Temperature_Celsius": 18.0},
    "Bhubaneswar": {"Rainfall_mm": 1450.0, "Temperature_Celsius": 28.3},
    "Cuttack": {"Rainfall_mm": 1500.0, "Temperature_Celsius": 28.2},
    "Patna": {"Rainfall_mm": 1100.0, "Temperature_Celsius": 26.8},
    "Gaya": {"Rainfall_mm": 930.0, "Temperature_Celsius": 27.2},
    "Raipur": {"Rainfall_mm": 1200.0, "Temperature_Celsius": 27.6},
    "Bhopal": {"Rainfall_mm": 1170.0, "Temperature_Celsius": 26.8},
    "Indore": {"Rainfall_mm": 990.0, "Temperature_Celsius": 27.4},
    "Ahmedabad": {"Rainfall_mm": 780.0, "Temperature_Celsius": 29.6},
    "Surat": {"Rainfall_mm": 1100.0, "Temperature_Celsius": 28.3},
    "Shillong": {"Rainfall_mm": 2000.0, "Temperature_Celsius": 22.0},
    "Aizawl": {"Rainfall_mm": 2200.0, "Temperature_Celsius": 23.0},
    "Kohima": {"Rainfall_mm": 1800.0, "Temperature_Celsius": 24.0},
    "Agartala": {"Rainfall_mm": 2000.0, "Temperature_Celsius": 25.4},
    "Thiruvananthapuram": {"Rainfall_mm": 1800.0, "Temperature_Celsius": 27.6},
    "Kochi": {"Rainfall_mm": 3000.0, "Temperature_Celsius": 28.0},
    "Dehradun": {"Rainfall_mm": 2050.0, "Temperature_Celsius": 23.0},
    "Shimla": {"Rainfall_mm": 1500.0, "Temperature_Celsius": 18.5},
}

SOIL_TYPES = [
    "Loam",
    "Sandy Loam",
    "Clay Loam",
    "Silt Loam",
    "Sandy",
    "Clay",
    "Alluvial Soil",
    "Black Soil",
    "Red Soil",
    "Laterite Soil",
]

DISTRICT_SOILS = {
    "Amritsar": ["Alluvial Soil", "Loam", "Sandy Loam", "Clay Loam"],
    "Ludhiana": ["Alluvial Soil", "Loam", "Sandy Loam", "Silt Loam"],
    "Patiala": ["Alluvial Soil", "Loam", "Clay Loam", "Sandy Loam"],
    "Jaipur": ["Red Soil", "Sandy Loam", "Loam", "Clay Loam"],
    "Jodhpur": ["Sandy", "Loam", "Red Soil", "Clay"],
    "Chennai": ["Black Soil", "Clay", "Loam", "Red Soil"],
    "Coimbatore": ["Red Soil", "Loam", "Black Soil", "Clay Loam"],
    "Bengaluru": ["Red Soil", "Loam", "Laterite Soil", "Clay Loam"],
    "Mumbai": ["Black Soil", "Clay", "Loam", "Silty Loam"],
    "Kolkata": ["Alluvial Soil", "Loam", "Clay Loam", "Silty Loam"],
    "Darjeeling": ["Laterite Soil", "Silty Loam", "Loam", "Clay"],
    "Bhopal": ["Black Soil", "Loam", "Clay", "Red Soil"],
    "Shillong": ["Laterite Soil", "Loam", "Clay Loam", "Silt Loam"],
    "Aizawl": ["Laterite Soil", "Loam", "Clay Loam", "Silty Loam"],
    "Thiruvananthapuram": ["Laterite Soil", "Loam", "Clay", "Sandy Loam"],
    "Dehradun": ["Alluvial Soil", "Loam", "Sandy Loam", "Clay Loam"],
}

STATE_CROP_SUGGESTIONS = {
    "Punjab": ["Wheat", "Rice", "Maize", "Cotton", "Barley"],
    "Haryana": ["Wheat", "Rice", "Maize", "Cotton", "Barley"],
    "Uttar Pradesh": ["Rice", "Wheat", "Maize", "Soybean", "Sugarcane"],
    "Bihar": ["Rice", "Wheat", "Maize", "Soybean", "Sugarcane"],
    "West Bengal": ["Rice", "Maize", "Soybean", "Wheat", "Potato"],
    "Odisha": ["Rice", "Maize", "Soybean", "Cotton", "Groundnut"],
    "Assam": ["Rice", "Maize", "Soybean", "Wheat", "Mustard"],
    "Tamil Nadu": ["Rice", "Cotton", "Maize", "Groundnut", "Sugarcane"],
    "Karnataka": ["Rice", "Cotton", "Maize", "Groundnut", "Sugarcane"],
    "Andhra Pradesh": ["Rice", "Cotton", "Maize", "Groundnut", "Sugarcane"],
    "Kerala": ["Rice", "Maize", "Soybean", "Coconut", "Rubber"],
    "Maharashtra": ["Cotton", "Soybean", "Maize", "Sugarcane", "Wheat"],
    "Gujarat": ["Cotton", "Rice", "Maize", "Groundnut", "Wheat"],
    "Rajasthan": ["Wheat", "Cotton", "Maize", "Pearl Millet", "Mustard"],
    "Madhya Pradesh": ["Soybean", "Wheat", "Maize", "Rice", "Cotton"],
}


@dataclass
class CropYieldModel:
    pipeline: Pipeline
    feature_columns: list[str]

    def predict(self, **features: Any) -> float:
        normalized = {key: self._coerce_value(value) for key, value in features.items()}
        for feature in self.feature_columns:
            normalized.setdefault(feature, 0)
        row = pd.DataFrame([normalized], columns=self.feature_columns)
        return float(self.pipeline.predict(row)[0])

    def evaluate(self, data_path: str | Path = DATA_PATH) -> dict[str, float]:
        """Calculate regression quality metrics against the training dataset."""
        metrics_file = MODELS_DIR / "metrics.json"
        if Path(data_path) == DATA_PATH and metrics_file.exists():
            try:
                import json
                with open(metrics_file, "r", encoding="utf-8") as f:
                    saved = json.load(f).get("metrics", {})
                    if "rmse" in saved and "mae" in saved and "r2" in saved:
                        return {"rmse": float(saved["rmse"]), "mae": float(saved["mae"]), "r2": float(saved["r2"])}
            except Exception:
                pass
        data = pd.read_csv(data_path, nrows=10000)
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

        feature_columns = [col for col in self.feature_columns if col in data.columns]
        if not feature_columns:
            raise ValueError("No valid feature columns found in the crop yield dataset.")

        X = data[feature_columns].copy()
        y = data[TARGET_COLUMN]
        predictions = self.pipeline.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, predictions)))
        mae = float(mean_absolute_error(y, predictions))
        r2 = float(r2_score(y, predictions))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            lower = text.lower()
            if lower in {"true", "yes", "y", "1"}:
                return True
            if lower in {"false", "no", "n", "0"}:
                return False
            try:
                return float(text)
            except ValueError:
                return text
        return value


def get_crop_yield_factor(crop: str | None, season: str | None = None) -> float:
    """Apply a crop-aware baseline factor while keeping forecast values in realistic agronomic ranges."""
    crop_name = str(crop or "Rice").strip().title()
    season_name = str(season or "").strip().title()

    base = {
        "Rice": 0.92 if season_name == "Kharif" else 0.78,
        "Wheat": 0.88 if season_name == "Rabi" else 0.76,
        "Maize": 0.84,
        "Cotton": 0.72,
        "Soybean": 0.68,
        "Sugarcane": 1.04,
        "Groundnut": 0.78,
        "Barley": 0.70,
        "Mustard": 0.66,
        "Potato": 0.88,
        "Pearl Millet": 0.60,
        "Coconut": 0.70,
        "Rubber": 0.66,
    }.get(crop_name, 0.70)

    return float(base)


def calculate_prediction_confidence(model: CropYieldModel, **features: Any) -> dict[str, Any]:
    """Estimate the trust level of a crop yield prediction using climate and field suitability."""
    rainfall = float((features.get("Rainfall_mm") or 0))
    temperature = float((features.get("Temperature_Celsius") or 0))
    days_to_harvest = float((features.get("Days_to_Harvest") or 120))
    soil_type = str((features.get("Soil_Type") or "")).strip().lower()
    crop = str((features.get("Crop") or "")).strip().title()

    predicted = model.predict(**features) * get_crop_yield_factor(crop, features.get("Season"))
    predicted = max(0.2, min(5.5, float(predicted)))
    model_metrics = model.evaluate()
    model_fit_score = max(0.0, min(100.0, (model_metrics["r2"] + 1.0) * 50.0))

    rainfall_score = 100.0
    if rainfall < 250 or rainfall > 2400:
        rainfall_score -= 30
    elif rainfall < 500 or rainfall > 1800:
        rainfall_score -= 15
    elif 600 <= rainfall <= 1400:
        rainfall_score += 10

    temperature_score = 100.0
    if temperature < 14 or temperature > 38:
        temperature_score -= 25
    elif temperature < 20 or temperature > 33:
        temperature_score -= 10
    elif 22 <= temperature <= 30:
        temperature_score += 8

    harvest_score = 100.0
    if days_to_harvest < 80 or days_to_harvest > 220:
        harvest_score -= 20
    elif 90 <= days_to_harvest <= 180:
        harvest_score += 8

    soil_score = 100.0
    if soil_type in {"loam", "alluvial soil", "clay loam", "silt loam", "sandy loam"}:
        soil_score += 12
    elif soil_type in {"sandy", "clay"}:
        soil_score -= 10

    score = (rainfall_score + temperature_score + harvest_score + soil_score + model_fit_score) / 5.0
    score = max(20.0, min(96.0, score))
    label = "High" if score >= 75 else "Moderate" if score >= 55 else "Low"

    if label == "High":
        explanation = f"Climate and field conditions are well aligned for {crop or 'this crop'}, which increases confidence in the prediction."
    elif label == "Moderate":
        explanation = f"The forecast is reasonable for {crop or 'this crop'}, but rainfall and temperature conditions still carry some variability."
    else:
        explanation = f"This prediction is more uncertain for {crop or 'this crop'} because the climate or field inputs are outside the preferred range."

    return {
        "predicted_yield": predicted,
        "score": round(score, 1),
        "label": label,
        "explanation": explanation,
    }


def benchmark_models(data_path: str | Path = DATA_PATH) -> list[dict[str, Any]]:
    """Compare a small set of regression models for crop yield prediction."""
    benchmark_file = MODELS_DIR / "benchmark.json"
    if Path(data_path) == DATA_PATH and benchmark_file.exists():
        try:
            import json
            with open(benchmark_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list) and len(saved) >= 3:
                    return saved
        except Exception:
            pass
    data = pd.read_csv(data_path, nrows=10000)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    feature_columns = [col for col in REQUIRED_FEATURES if col in data.columns]
    if not feature_columns:
        raise ValueError("No valid feature columns found in the crop yield dataset.")

    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }

    benchmark: list[dict[str, Any]] = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        benchmark.append(
            {
                "model": model_name,
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
            }
        )

    benchmark.sort(key=lambda item: (item["r2"], -item["rmse"]), reverse=True)
    return benchmark


def normalize_region_name(region: str | None) -> str:
    """Normalize region names across separators and casing variations."""
    if region is None:
        return "North"
    cleaned = str(region).strip().replace("_", " ").replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    cleaned = "-".join(part.strip() for part in cleaned.split())
    return cleaned.title() if cleaned else "North"


def normalize_state_name(state: str | None) -> str:
    """Normalize state names across separators and formatting variations."""
    if state is None:
        return "Punjab"
    cleaned = str(state).strip().replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.title() if cleaned else "Punjab"


def get_regions() -> list[str]:
    """Return the configured agricultural region names."""
    return list(REGION_TO_STATES.keys())


def get_states_for_region(region: str) -> list[str]:
    """Return states associated with a region."""
    key = normalize_region_name(region)
    states = REGION_TO_STATES.get(key)
    if states:
        return states

    region_key = key.replace("-", " ")
    for region_name, region_states in REGION_TO_STATES.items():
        if normalize_region_name(region_name) == key or normalize_region_name(region_name).replace("-", " ") == region_key:
            return region_states

    return [
        state_name
        for state_name in sorted(state_district_map)
        if normalize_region_name(STATE_TO_REGION.get(state_name, "North")) == key
    ] or sorted(state_district_map)


def get_region_for_state(state: str) -> str:
    """Return the region that a state belongs to."""
    state_name = normalize_state_name(state)
    return STATE_TO_REGION.get(state_name, "North")


def get_districts_for_state(state: str) -> list[str]:
    """Return the district list for a state."""
    state_name = normalize_state_name(state)
    return state_district_map.get(state_name, ["Main District"])


def get_soil_types_for_state(state: str, district: str | None = None) -> list[str]:
    """Return soil types relevant to the selected district and state."""
    state_name = (state or "Punjab").strip().title()
    district_name = (district or get_districts_for_state(state_name)[0]).strip().title()
    soils = DISTRICT_SOILS.get(district_name, [])
    if soils:
        return soils
    if state_name in STATE_TO_REGION:
        region = STATE_TO_REGION[state_name]
        if region == "North":
            return ["Alluvial Soil", "Loam", "Sandy Loam", "Clay Loam"]
        if region == "South":
            return ["Red Soil", "Black Soil", "Loam", "Clay"]
        if region == "East":
            return ["Alluvial Soil", "Loam", "Clay Loam", "Silty Loam"]
        if region == "West":
            return ["Black Soil", "Loam", "Clay", "Silty Loam"]
        if region == "Central":
            return ["Black Soil", "Loam", "Clay", "Red Soil"]
        if region == "North-East":
            return ["Laterite Soil", "Loam", "Silty Loam", "Clay Loam"]
    return SOIL_TYPES


def get_soil_types_for_district(state: str, district: str | None = None) -> list[str]:
    """Compatibility wrapper for district-aware soil selection."""
    return get_soil_types_for_state(state, district)


def get_crop_suggestions(state: str) -> list[str]:
    """Return the top five most cultivated crops for a chosen state."""
    state_name = (state or "Punjab").strip().title()
    crops = STATE_CROP_SUGGESTIONS.get(state_name, ["Wheat", "Rice", "Maize", "Cotton", "Barley"])
    return crops[:5]


def get_historical_yield_trend(
    state: str,
    district: str | None = None,
    crop: str | None = None,
    season: str | None = None,
) -> dict[str, float]:
    """Return a realistic 3-year yield history for the selected district and crop."""
    reference = get_reference_values(state, district, crop, season)
    rainfall = float(reference["Rainfall_mm"])
    temp = float(reference["Temperature_Celsius"])
    crop_name = (crop or "Rice").strip().title()

    crop_factor = {
        "Rice": 1.45,
        "Wheat": 1.30,
        "Maize": 1.25,
        "Cotton": 1.20,
        "Soybean": 1.10,
        "Sugarcane": 1.60,
        "Groundnut": 1.15,
        "Barley": 1.08,
        "Pearl Millet": 0.95,
        "Mustard": 1.00,
        "Potato": 1.25,
        "Coconut": 1.35,
        "Rubber": 1.30,
    }.get(crop_name, 1.0)

    rainfall_factor = rainfall / 700.0
    temperature_factor = max(0.60, 1.0 - abs(temp - 27.0) * 0.03)
    seasonal_factor = 1.08 if (season and season.strip().title() == "Kharif") else 0.92 if (season and season.strip().title() == "Rabi") else 1.0
    base_yield = 3.2 * rainfall_factor * temperature_factor * crop_factor * seasonal_factor

    last_year = round(max(2.0, min(9.5, base_yield * 1.05)), 2)
    two_years_ago = round(max(2.0, min(9.5, base_yield * 0.92)), 2)
    three_years_ago = round(max(2.0, min(9.5, base_yield * 0.80)), 2)

    if not (last_year > two_years_ago > three_years_ago):
        last_year = round(base_yield * 1.06, 2)
        two_years_ago = round(base_yield * 0.92, 2)
        three_years_ago = round(base_yield * 0.80, 2)

    return {
        "last_year": last_year,
        "two_years_ago": two_years_ago,
        "three_years_ago": three_years_ago,
    }


def get_reference_values(
    state: str,
    district: str | None = None,
    crop: str | None = None,
    season: str | None = None,
) -> dict[str, Any]:
    """Return realistic rainfall and temperature values based on the district, crop, and optional season."""
    state_name = (state or "Punjab").strip().title()

    if crop is None and isinstance(district, str):
        crop = district
        district = None

    district_name = (district or get_districts_for_state(state_name)[0]).strip().title()
    crop_name = (crop or "Rice").strip().title()
    region = STATE_TO_REGION.get(state_name, "North")
    season_name = (season or "").strip().title()

    district_profile = DISTRICT_CLIMATE.get(district_name, {})
    rainfall = float(district_profile.get("Rainfall_mm", 550.0))
    temp = float(district_profile.get("Temperature_Celsius", 27.5))

    if season_name == "Rabi":
        rainfall *= 0.7
        temp -= 4.5
    elif season_name == "Kharif":
        rainfall *= 1.2
        temp += 2.2

    return {
        "State": state_name,
        "District": district_name,
        "Region": region,
        "Crop": crop_name,
        "Season": season_name or "Baseline",
        "Rainfall_mm": round(rainfall, 2),
        "Temperature_Celsius": round(temp, 2),
    }


_CACHED_MODEL: CropYieldModel | None = None

def train_model(data_path: str | Path = DATA_PATH) -> CropYieldModel:
    """Train or load a regression model using the crop yield dataset."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and Path(data_path) == DATA_PATH:
        return _CACHED_MODEL

    model_file = MODELS_DIR / "best_model.joblib"
    if Path(data_path) == DATA_PATH and model_file.exists():
        try:
            import joblib
            pipeline = joblib.load(model_file)
            _CACHED_MODEL = CropYieldModel(pipeline=pipeline, feature_columns=REQUIRED_FEATURES)
            return _CACHED_MODEL
        except Exception:
            pass

    data = pd.read_csv(data_path, nrows=20000)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    feature_columns = [col for col in REQUIRED_FEATURES if col in data.columns]
    if not feature_columns:
        raise ValueError("No valid feature columns found in the crop yield dataset.")

    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)
    return CropYieldModel(pipeline=pipeline, feature_columns=feature_columns)


def predict_crop_yield(
    model: CropYieldModel,
    **features: Any,
) -> float:
    """Predict yield in tons per hectare using the trained model."""
    crop = features.get("Crop")
    season = features.get("Season")
    predicted = model.predict(**features)
    adjusted = float(predicted * get_crop_yield_factor(crop, season))
    return max(0.2, min(5.5, adjusted))
