"""
Video-to-Signal Controller Pipeline
Processes traffic video, detects/counts vehicles, and predicts optimal green light timing.
"""

import os
import sys
from layer1_yolo.detector import process_video
from layer2_ml.predict import predict_green_time_class
import json
from datetime import datetime


def process_traffic_video(video_path, output_video_path=None, rain=0):
    """
    Complete pipeline: video → vehicle detection → signal prediction

    Args:
        video_path        : path to traffic video
        output_video_path : optional path to save annotated video
        rain              : whether it's raining (0 or 1)

    Returns:
        dict with video analysis and signal predictions
    """
    print(f"\n{'='*60}")
    print(f"TRAFFIC SIGNAL CONTROLLER - VIDEO ANALYSIS")
    print(f"{'='*60}\n")
    print(f"📹 Processing video: {video_path}")

    # ── Step 1: Process video ──────────────────────────────────────────────
    # NOTE: aggregation_frames is no longer used — the new detector always
    # uses 1-second windows internally.  The parameter is accepted but ignored
    # so existing call-sites don't break.
    video_result = process_video(video_path, output_video_path)

    print(f"✓ Video processed: {video_result['total_frames']} frames")
    print(f"  Duration : {video_result['duration']:.2f}s @ {video_result['fps']} FPS")
    print(f"  Windows  : {len(video_result['aggregated_counts'])} × 1-second windows\n")

    unique = video_result["unique_totals"]
    print(f"  Unique vehicles tracked:")
    print(f"    Cars     : {unique['car_count']}")
    print(f"    Buses    : {unique['bus_truck_count']}")
    print(f"    Bikes    : {unique['bike_count']}\n")

    # ── Step 2: Predict green time for each 1-second window ───────────────
    predictions = []

    for agg in video_result["aggregated_counts"]:
        car_count  = agg["car_count"]
        bus_count  = agg["bus_truck_count"]
        bike_count = agg["bike_count"]

        pred = predict_green_time_class(car_count, bus_count, bike_count, rain=rain)

        # "start_frame" / "end_frame" were removed from the new window dict.
        # We reconstruct an approximate range from fps and window index.
        fps        = video_result["fps"]
        win_idx    = agg["window"]
        start_sec  = (win_idx - 1)          # window N covers second N-1 → N
        end_sec    = win_idx
        frame_range = f"~{start_sec*fps}-{end_sec*fps}"

        prediction_data = {
            "window":      win_idx,
            "frame_range": frame_range,       # approximate — exact frames removed in v2
            "timestamp":   agg["timestamp"],
            "vehicle_counts": {
                "cars":  car_count,
                "buses": bus_count,
                "bikes": bike_count,
                "total": car_count + bus_count + bike_count,
            },
            "prediction": {
                "green_time_seconds": pred["predicted_green_time"],
                "confidence":         round(pred["confidence"], 4),
                "probabilities": {
                    "30s":  round(pred["probabilities"].get("class_30s",  0.0), 4),
                    "60s":  round(pred["probabilities"].get("class_60s",  0.0), 4),
                    "90s":  round(pred["probabilities"].get("class_90s",  0.0), 4),
                    "120s": round(pred["probabilities"].get("class_120s", 0.0), 4),
                },
            },
        }
        predictions.append(prediction_data)

    # ── Step 3: Also predict using cumulative unique counts ───────────────
    # This gives a single "whole-video" recommendation — more stable than
    # per-window predictions which can fluctuate with partial detections.
    unique_pred = predict_green_time_class(
        unique["car_count"],
        unique["bus_truck_count"],
        unique["bike_count"],
        rain=rain,
    )

    # ── Step 4: Print summary ─────────────────────────────────────────────
    if predictions:
        green_times    = [p["prediction"]["green_time_seconds"] for p in predictions]
        avg_green_time = sum(green_times) / len(green_times)

        print(f"{'='*60}")
        print(f"PREDICTIONS BY 1-SECOND WINDOW")
        print(f"{'='*60}\n")

        for pred in predictions[:10]:
            print(f"Window {pred['window']} (t={pred['timestamp']:.2f}s)")
            print(f"  Vehicles : {pred['vehicle_counts']['cars']} cars, "
                  f"{pred['vehicle_counts']['buses']} buses, "
                  f"{pred['vehicle_counts']['bikes']} bikes")
            print(f"  → Green time: {pred['prediction']['green_time_seconds']}s "
                  f"(confidence: {pred['prediction']['confidence']:.1%})")
            probs = pred["prediction"]["probabilities"]
            print(f"     30s={probs['30s']:.1%}  60s={probs['60s']:.1%}  "
                  f"90s={probs['90s']:.1%}  120s={probs['120s']:.1%}\n")

        if len(predictions) > 10:
            print(f"   ... and {len(predictions) - 10} more windows\n")

        print(f"{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}\n")
        print(f"Per-window average green time : {avg_green_time:.1f}s")
        print(f"Per-window range              : {min(green_times)}s – {max(green_times)}s")
        print(f"Most common per-window        : {max(set(green_times), key=green_times.count)}s")
        print(f"Whole-video recommendation    : {unique_pred['predicted_green_time']}s "
              f"(confidence: {unique_pred['confidence']:.1%})")
        print(f"Rain condition                : {'Yes' if rain else 'No'}\n")

    else:
        avg_green_time = 0
        green_times    = []
        print(" No aggregation windows produced — video may be too short.")

    # ── Build result dict ─────────────────────────────────────────────────
    result = {
        "timestamp":      datetime.now().isoformat(),
        "video_analysis": {
            "total_frames":      video_result["total_frames"],
            "fps":               video_result["fps"],
            "duration":          video_result["duration"],
            "unique_totals":     unique,
            "aggregated_counts": video_result["aggregated_counts"],
        },
        "predictions": predictions,
        "whole_video_prediction": {
            "green_time_seconds": unique_pred["predicted_green_time"],
            "confidence":         round(unique_pred["confidence"], 4),
            "based_on":           unique,
        },
        "summary": {
            "total_windows":          len(predictions),
            "average_green_time":     round(avg_green_time, 1),
            "min_green_time":         min(green_times) if green_times else 0,
            "max_green_time":         max(green_times) if green_times else 0,
            "recommended_green_time": unique_pred["predicted_green_time"],
            "rain_condition":         bool(rain),
        },
    }

    return result


def save_results(result, output_json_path):
    """Save analysis results to JSON."""
    # video_result contains numpy arrays in frame_logs — strip them for JSON
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✓ Results saved to: {output_json_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python traffic_pipeline.py <video_path> [output_video_path] [rain=0]")
        print("\nExamples:")
        print("  python traffic_pipeline.py traffic.mp4")
        print("  python traffic_pipeline.py traffic.mp4 output.mp4 1  # with rain")
        sys.exit(1)

    video_path   = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    rain         = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if not os.path.exists(video_path):
        print(f" Video not found: {video_path}")
        sys.exit(1)

    try:
        result = process_traffic_video(video_path, output_video, rain=rain)

        results_dir = "layer3_sumo/results"
        os.makedirs(results_dir, exist_ok=True)
        base_name   = os.path.splitext(os.path.basename(video_path))[0]
        output_json = os.path.join(results_dir, f"{base_name}_predictions.json")
        save_results(result, output_json)

        print("✓ Pipeline complete!")

    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)