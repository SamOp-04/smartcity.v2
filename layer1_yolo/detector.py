"""
YOLOv8-based vehicle detection and tracking module.

This module provides real-time vehicle detection using YOLOv8 and
centroid-based tracking to count unique vehicles in video streams.
"""

from typing import Dict, List, Optional, Generator, Any, Union
from ultralytics import YOLO
import cv2
import numpy as np
import os

from utils.constants import (
    VEHICLE_GROUPS,
    YOLO_CLASS_MAP,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_SKIP,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_DIST,
    DEFAULT_MAX_FRAMES_MISSING,
)
from utils.logging_config import get_logger
from utils.validation import (
    validate_confidence_threshold,
    validate_frame_parameters,
    validate_image,
    ValidationError,
)

logger = get_logger(__name__)

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    """
    Load and cache the YOLOv8 model.

    Returns:
        YOLO: Loaded YOLOv8 model instance
    """
    global _model
    if _model is None:
        model_paths = [
            "yolov8n.pt",
            os.path.join(os.path.dirname(__file__), "yolov8n.pt"),
            os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt"),
            os.path.join(os.path.dirname(__file__), "..", "layer4_dashboard", "yolov8n.pt"),
        ]
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        if model_path is None:
            model_path = "yolov8n.pt"
        logger.info(f"Loading YOLO model from: {model_path}")
        _model = YOLO(model_path)
    return _model


def detect_vehicles(
    image_source: Union[str, np.ndarray],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Detect and count vehicles in a single image.

    Args:
        image_source: Image path or numpy array
        confidence_threshold: Minimum detection confidence (0.0-1.0)

    Returns:
        Dict containing:
            - car_count: Number of cars detected
            - bus_truck_count: Number of buses/trucks detected
            - bike_count: Number of bikes detected
            - detections: List of detection details
            - annotated_frame: Image with detection boxes drawn
    """
    confidence_threshold = validate_confidence_threshold(confidence_threshold)
    model = get_model()
    results = model(image_source, verbose=False)[0]

    counts: Dict[str, int] = {"car_count": 0, "bus_truck_count": 0, "bike_count": 0}
    detections: List[Dict[str, Any]] = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < confidence_threshold:
            continue
        if cls_id not in YOLO_CLASS_MAP:
            continue

        group: Optional[str] = None
        if cls_id in VEHICLE_GROUPS["CAR"]:
            counts["car_count"] += 1
            group = "CAR"
        elif cls_id in VEHICLE_GROUPS["BUS_TRUCK"]:
            counts["bus_truck_count"] += 1
            group = "BUS_TRUCK"
        elif cls_id in VEHICLE_GROUPS["BIKE"]:
            counts["bike_count"] += 1
            group = "BIKE"
        else:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class": YOLO_CLASS_MAP[cls_id],
            "group": group,
            "confidence": round(conf, 3),
        })

    return {**counts, "detections": detections, "annotated_frame": results.plot()}


class SimpleTracker:
    """
    Centroid-based vehicle tracker for counting unique vehicles.

    Uses Euclidean distance between centroids to match detections
    across frames and maintain consistent track IDs.
    """

    def __init__(
        self,
        max_dist: int = DEFAULT_MAX_DIST,
        max_frames_missing: int = DEFAULT_MAX_FRAMES_MISSING,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            max_dist: Maximum pixel distance to match existing tracks
            max_frames_missing: Frames before a track is deleted
        """
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id: int = 0
        self.max_dist: int = max_dist
        self.max_frames_missing: int = max_frames_missing
        self.total_unique: Dict[str, int] = {"CAR": 0, "BUS_TRUCK": 0, "BIKE": 0}

    def _centroid(self, bbox: List[float]) -> tuple[float, float]:
        """Calculate centroid from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _dist(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def update(
        self,
        detections: List[Dict[str, Any]],
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detection dicts with 'bbox' and 'group' keys

        Returns:
            Tuple of (current_counts, unique_counts) dictionaries
        """
        matched_ids: set[int] = set()

        for det in detections:
            centroid = self._centroid(det["bbox"])
            group = det["group"]
            best_id: Optional[int] = None
            best_dist: float = self.max_dist

            for tid, td in self.tracks.items():
                if td["class"] != group:
                    continue
                d = self._dist(centroid, td["centroid"])
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id]["centroid"] = centroid
                self.tracks[best_id]["frames_missing"] = 0
                matched_ids.add(best_id)
            else:
                self.tracks[self.next_track_id] = {
                    "centroid": centroid,
                    "class": group,
                    "frames_missing": 0,
                }
                self.total_unique[group] += 1
                matched_ids.add(self.next_track_id)
                self.next_track_id += 1

        for tid in list(self.tracks):
            if tid not in matched_ids:
                self.tracks[tid]["frames_missing"] += 1
                if self.tracks[tid]["frames_missing"] > self.max_frames_missing:
                    del self.tracks[tid]

        current_counts: Dict[str, int] = {
            "car_count": 0,
            "bus_truck_count": 0,
            "bike_count": 0,
        }
        for td in self.tracks.values():
            if td["class"] == "CAR":
                current_counts["car_count"] += 1
            elif td["class"] == "BUS_TRUCK":
                current_counts["bus_truck_count"] += 1
            elif td["class"] == "BIKE":
                current_counts["bike_count"] += 1

        unique_counts: Dict[str, int] = {
            "car_count": self.total_unique["CAR"],
            "bus_truck_count": self.total_unique["BUS_TRUCK"],
            "bike_count": self.total_unique["BIKE"],
        }
        return current_counts, unique_counts


def process_video_live(
    video_path: str,
    output_path: Optional[str] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields detection results frame by frame.

    Args:
        video_path: Path to input video file
        output_path: Optional path for annotated output video
        confidence_threshold: Minimum detection confidence (0.0-1.0)
        frame_skip: Process every Nth frame (default: 5)
        max_frames: Maximum frames to process (default: 300)

    Yields:
        Dict containing frame data, counts, progress, and done status
    """
    frame_skip, max_frames = validate_frame_parameters(frame_skip, max_frames)
    confidence_threshold = validate_confidence_threshold(confidence_threshold)

    model = get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    effective_total = min(total_frames, max_frames)
    agg_frames = max(fps, 1)

    tracker = SimpleTracker(
        max_dist=DEFAULT_MAX_DIST,
        max_frames_missing=DEFAULT_MAX_FRAMES_MISSING,
    )

    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_logs: List[Dict[str, Any]] = []
    window_counts: List[Dict[str, Any]] = []
    frame_count = 0
    window_peak: Dict[str, int] = {"car_count": 0, "bus_truck_count": 0, "bike_count": 0}

    last_annotated: Optional[np.ndarray] = None
    last_current: Dict[str, int] = {"car_count": 0, "bus_truck_count": 0, "bike_count": 0}
    last_unique: Dict[str, int] = {"car_count": 0, "bus_truck_count": 0, "bike_count": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count > max_frames:
                break

            timestamp = frame_count / fps
            progress = frame_count / max(effective_total, 1)

            if frame_count % frame_skip == 0:
                results = model(frame, verbose=False)[0]
                detections: List[Dict[str, Any]] = []

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < confidence_threshold:
                        continue
                    if cls_id not in YOLO_CLASS_MAP:
                        continue
                    if cls_id == 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    group: Optional[str] = None
                    if cls_id in VEHICLE_GROUPS["CAR"]:
                        group = "CAR"
                    elif cls_id in VEHICLE_GROUPS["BUS_TRUCK"]:
                        group = "BUS_TRUCK"
                    elif cls_id in VEHICLE_GROUPS["BIKE"]:
                        group = "BIKE"
                    else:
                        continue

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": YOLO_CLASS_MAP[cls_id],
                        "group": group,
                        "confidence": round(conf, 3),
                    })

                current_counts, unique_counts = tracker.update(detections)

                annotated = results.plot()

                for tid, td in tracker.tracks.items():
                    cx, cy = int(td["centroid"][0]), int(td["centroid"][1])
                    cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(
                        annotated,
                        f"#{tid}",
                        (cx - 10, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

                cv2.putText(
                    annotated,
                    f"Cars:{current_counts['car_count']}  "
                    f"Buses:{current_counts['bus_truck_count']}  "
                    f"Bikes:{current_counts['bike_count']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )

                last_annotated = annotated
                last_current = current_counts
                last_unique = unique_counts

                frame_logs.append({
                    "frame": frame_count,
                    "timestamp": timestamp,
                    **current_counts,
                    "detections": len(detections),
                })

                for k in ("car_count", "bus_truck_count", "bike_count"):
                    window_peak[k] = max(window_peak[k], current_counts[k])

                if frame_count % agg_frames == 0:
                    window_counts.append({
                        "window": len(window_counts) + 1,
                        "timestamp": timestamp,
                        "car_count": window_peak["car_count"],
                        "bus_truck_count": window_peak["bus_truck_count"],
                        "bike_count": window_peak["bike_count"],
                    })
                    window_peak = {"car_count": 0, "bus_truck_count": 0, "bike_count": 0}

            if last_annotated is not None:
                if writer:
                    writer.write(last_annotated)
                yield {
                    "annotated": last_annotated,
                    "current": last_current,
                    "unique": last_unique,
                    "frame_number": frame_count,
                    "total_frames": effective_total,
                    "timestamp": timestamp,
                    "fps": fps,
                    "progress": progress,
                    "window_counts": list(window_counts),
                    "done": False,
                }

    finally:
        cap.release()
        if writer:
            writer.release()

    # Flush last partial window
    if any(window_peak.values()):
        window_counts.append({
            "window": len(window_counts) + 1,
            "timestamp": frame_count / fps,
            **window_peak,
        })

    logger.info(
        f"Video processing complete: {frame_count} frames, "
        f"{tracker.total_unique} unique vehicles"
    )

    yield {
        "annotated": last_annotated,
        "current": last_current,
        "unique": last_unique,
        "frame_number": frame_count,
        "total_frames": effective_total,
        "timestamp": frame_count / fps,
        "fps": fps,
        "progress": 1.0,
        "window_counts": window_counts,
        "frame_logs": frame_logs,
        "unique_totals": {
            "car_count": tracker.total_unique["CAR"],
            "bus_truck_count": tracker.total_unique["BUS_TRUCK"],
            "bike_count": tracker.total_unique["BIKE"],
        },
        "done": True,
    }


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    aggregation_frames: Optional[int] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Non-generator batch version that processes full video.

    Args:
        video_path: Path to input video file
        output_path: Optional path for annotated output video
        aggregation_frames: Deprecated, kept for compatibility
        confidence_threshold: Minimum detection confidence (0.0-1.0)

    Returns:
        Dict containing frame logs, aggregated counts, and video metadata
    """
    results_list = list(
        process_video_live(
            video_path,
            output_path=output_path,
            confidence_threshold=confidence_threshold,
            frame_skip=1,
            max_frames=999999,
        )
    )
    final = results_list[-1]
    fps = final["fps"]
    return {
        "frame_logs": final.get("frame_logs", []),
        "aggregated_counts": final.get("window_counts", []),
        "total_frames": final["total_frames"],
        "fps": fps,
        "duration": final["total_frames"] / fps if fps else 0,
        "video_path": video_path,
        "output_path": output_path,
        "unique_vehicles_tracked": final["unique_totals"],
        "unique_totals": final["unique_totals"],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m layer1_yolo.detector <video_path> [output_path]")
        print(f"YOLOv8 loaded: {get_model().model_name}")
        sys.exit(0)
    if sys.argv[1] == "--image":
        r = detect_vehicles(sys.argv[2])
        print(f"Cars:{r['car_count']} Buses:{r['bus_truck_count']} Bikes:{r['bike_count']}")
    else:
        result = process_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        u = result["unique_totals"]
        print(f"Done - Cars:{u['car_count']} Buses:{u['bus_truck_count']} Bikes:{u['bike_count']}")
