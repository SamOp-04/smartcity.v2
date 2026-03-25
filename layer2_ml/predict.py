"""
XGBoost-based green signal time prediction module.

This module predicts optimal traffic signal green time duration
based on vehicle counts and weather conditions.
"""

from typing import Dict, Any, Optional
import joblib
import numpy as np
import os

from utils.logging_config import get_logger
from utils.validation import validate_vehicle_counts, validate_rain
from utils.constants import GREEN_TIME_CLASSES, MIN_GREEN_TIME, MAX_GREEN_TIME

logger = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_green_time.joblib")

_model: Optional[Any] = None

# Map class indices to green time values
CLASS_MAP: Dict[int, int] = {
    0: GREEN_TIME_CLASSES[0],  # 30
    1: GREEN_TIME_CLASSES[1],  # 60
    2: GREEN_TIME_CLASSES[2],  # 90
    3: GREEN_TIME_CLASSES[3],  # 120
}


def load_model() -> Any:
    """
    Load and cache the XGBoost model.

    Returns:
        Loaded XGBoost classifier model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python -m layer2_ml.train_model' first."
            )
        logger.info(f"Loading XGBoost model from: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_green_time_class(
    car_count: int,
    bus_truck_count: int,
    bike_count: int,
    rain: int = 0,
) -> Dict[str, Any]:
    """
    Predict optimal green signal time class with detailed probabilities.

    Args:
        car_count: Number of cars detected
        bus_truck_count: Number of buses/trucks detected
        bike_count: Number of bikes detected
        rain: Rain condition indicator (0 or 1)

    Returns:
        Dict containing:
            - predicted_class_index: Class index (0-3)
            - predicted_green_time: Green time in seconds (30/60/90/120)
            - probabilities: Dict of probabilities for each class
            - confidence: Confidence score for predicted class
    """
    # Validate inputs
    car_count, bus_truck_count, bike_count = validate_vehicle_counts(
        car_count, bus_truck_count, bike_count
    )
    rain = validate_rain(rain)

    model = load_model()
    X = np.array([[car_count, bus_truck_count, bike_count, rain]])

    # Get prediction and probabilities
    class_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    # Ensure class_idx is valid
    if class_idx not in CLASS_MAP:
        logger.warning(f"Invalid class index {class_idx}, defaulting to 0")
        class_idx = 0

    class_label = CLASS_MAP[class_idx]

    # Create probability dict
    probabilities: Dict[str, float] = {}
    for i in range(len(proba)):
        if i in CLASS_MAP:
            probabilities[f"class_{CLASS_MAP[i]}s"] = float(proba[i])

    logger.debug(
        f"Prediction: cars={car_count}, buses={bus_truck_count}, "
        f"bikes={bike_count}, rain={rain} -> {class_label}s (conf: {proba[class_idx]:.2f})"
    )

    return {
        "predicted_class_index": class_idx,
        "predicted_green_time": class_label,
        "probabilities": probabilities,
        "confidence": float(proba[class_idx]),
    }


def predict_green_time(
    car_count: int,
    bus_truck_count: int,
    bike_count: int,
    rain: int = 0,
) -> int:
    """
    Simple wrapper that returns only the predicted green time value.

    Args:
        car_count: Number of cars detected
        bus_truck_count: Number of buses/trucks detected
        bike_count: Number of bikes detected
        rain: Rain condition indicator (0 or 1)

    Returns:
        Predicted green time in seconds (30, 60, 90, or 120)
    """
    result = predict_green_time_class(car_count, bus_truck_count, bike_count, rain)
    return result["predicted_green_time"]


def estimate_green_time_formula(
    car_count: int,
    bus_truck_count: int,
    bike_count: int,
    rain: int = 0,
) -> int:
    """
    Fallback formula-based estimation when ML model is unavailable.

    Uses weighted sum: green = 2*cars + 3*buses + 1*bikes + 5*rain

    Args:
        car_count: Number of cars detected
        bus_truck_count: Number of buses/trucks detected
        bike_count: Number of bikes detected
        rain: Rain condition indicator (0 or 1)

    Returns:
        Estimated green time clamped to MIN_GREEN_TIME-MAX_GREEN_TIME
    """
    car_count, bus_truck_count, bike_count = validate_vehicle_counts(
        car_count, bus_truck_count, bike_count
    )
    rain = validate_rain(rain)

    raw_time = car_count * 2 + bus_truck_count * 3 + bike_count * 1 + rain * 5
    clamped = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, raw_time))

    # Round to nearest class
    for green_class in sorted(GREEN_TIME_CLASSES):
        if clamped <= green_class:
            return green_class
    return GREEN_TIME_CLASSES[-1]


if __name__ == "__main__":
    # Test examples
    print("Testing ML Prediction Module")
    print("-" * 40)

    result = predict_green_time_class(10, 3, 5, rain=0)
    print(f"Input: 10 cars, 3 buses, 5 bikes, no rain")
    print(f"Predicted green time: {result['predicted_green_time']}s")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")

    print()

    result_rain = predict_green_time_class(10, 3, 5, rain=1)
    print(f"Input: 10 cars, 3 buses, 5 bikes, with rain")
    print(f"Predicted green time: {result_rain['predicted_green_time']}s")
    print(f"Confidence: {result_rain['confidence']:.4f}")
