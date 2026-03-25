"""Centralized constants for the traffic optimization system."""

from typing import Dict, List, Tuple

# COCO class IDs relevant to traffic
# person (0) deliberately NOT included — excluded at detection level
YOLO_CLASS_MAP: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Road vehicles only — person excluded entirely
VEHICLE_GROUPS: Dict[str, List[int]] = {
    "CAR":       [2],       # car
    "BUS_TRUCK": [5, 7],    # bus + truck
    "BIKE":      [1, 3],    # bicycle + motorcycle only
}

# Feature names used by the ML model
ML_FEATURES: List[str] = ["car_count", "bus_truck_count", "bike_count", "rain"]

# Green time classification classes (seconds)
GREEN_TIME_CLASSES: List[int] = [30, 60, 90, 120]
CLASS_BOUNDARIES: Dict[int, Tuple[int, float]] = {
    30:  (0,   45),
    60:  (45,  75),
    90:  (75,  105),
    120: (105, float("inf")),
}

# Signal timing constraints
MIN_GREEN_TIME: int = 10
MAX_GREEN_TIME: int = 120
YELLOW_TIME: int = 3

# Detection defaults
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
DEFAULT_FRAME_SKIP: int = 5
DEFAULT_MAX_FRAMES: int = 300

# Tracker defaults
DEFAULT_MAX_DIST: int = 80
DEFAULT_MAX_FRAMES_MISSING: int = 5

# SUMO configuration
SUMO_SIM_DURATION: int = 600
SUMO_EDGES: Dict[str, List[str]] = {
    "NS": ["north_in", "south_in"],
    "EW": ["east_in", "west_in"],
}
