"""Input validation utilities for the traffic optimization system."""

from typing import Union, Optional
import numpy as np

from utils.constants import MIN_GREEN_TIME, MAX_GREEN_TIME, GREEN_TIME_CLASSES


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_vehicle_counts(
    car_count: int,
    bus_truck_count: int,
    bike_count: int,
    max_car: int = 500,
    max_bus: int = 100,
    max_bike: int = 500,
) -> tuple[int, int, int]:
    """
    Validate and clamp vehicle counts to reasonable ranges.

    Args:
        car_count: Number of cars detected
        bus_truck_count: Number of buses/trucks detected
        bike_count: Number of bikes detected
        max_car: Maximum reasonable car count
        max_bus: Maximum reasonable bus/truck count
        max_bike: Maximum reasonable bike count

    Returns:
        Tuple of validated (car_count, bus_truck_count, bike_count)

    Raises:
        ValidationError: If counts are negative
    """
    if any(c < 0 for c in [car_count, bus_truck_count, bike_count]):
        raise ValidationError("Vehicle counts cannot be negative")

    return (
        min(int(car_count), max_car),
        min(int(bus_truck_count), max_bus),
        min(int(bike_count), max_bike),
    )


def validate_rain(rain: Union[int, bool]) -> int:
    """
    Validate rain parameter.

    Args:
        rain: Rain indicator (0/1 or True/False)

    Returns:
        Validated rain value (0 or 1)
    """
    return 1 if rain else 0


def validate_confidence_threshold(threshold: float) -> float:
    """
    Validate detection confidence threshold.

    Args:
        threshold: Confidence threshold (0.0 to 1.0)

    Returns:
        Clamped threshold value
    """
    return max(0.0, min(1.0, float(threshold)))


def validate_green_time(green_time: int) -> int:
    """
    Validate and clamp green time to allowed range.

    Args:
        green_time: Proposed green time in seconds

    Returns:
        Clamped green time within MIN_GREEN_TIME and MAX_GREEN_TIME
    """
    return max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, int(green_time)))


def validate_frame_parameters(
    frame_skip: int = 5,
    max_frames: int = 300,
) -> tuple[int, int]:
    """
    Validate video processing frame parameters.

    Args:
        frame_skip: Process every Nth frame
        max_frames: Maximum frames to process

    Returns:
        Validated (frame_skip, max_frames)
    """
    frame_skip = max(1, min(30, int(frame_skip)))
    max_frames = max(1, int(max_frames))
    return frame_skip, max_frames


def validate_image(image: np.ndarray) -> np.ndarray:
    """
    Validate that input is a proper image array.

    Args:
        image: Input image as numpy array

    Returns:
        Validated image array

    Raises:
        ValidationError: If image is invalid
    """
    if image is None:
        raise ValidationError("Image cannot be None")

    if not isinstance(image, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(image)}")

    if len(image.shape) < 2:
        raise ValidationError(f"Invalid image dimensions: {image.shape}")

    return image
