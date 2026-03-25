"""
SUMO simulation comparison module.

Runs both fixed-time and adaptive signal control simulations
and produces comparison metrics.
"""

from typing import Dict, Any, Optional
import json
import os

from layer3_sumo.run_fixed import run_fixed_simulation, SUMOConnectionError
from layer3_sumo.run_adaptive import run_adaptive_simulation
from utils.logging_config import get_logger

logger = get_logger(__name__)


def compare(
    rain: int = 0,
    gui: bool = False,
    fixed_green_time: int = 42,
) -> Dict[str, Any]:
    """
    Run both simulations and produce comparison metrics.

    Args:
        rain: Rain condition indicator (0 or 1)
        gui: Whether to use SUMO GUI
        fixed_green_time: Green time for fixed simulation baseline

    Returns:
        Dict containing fixed results, adaptive results, and improvement percentages

    Raises:
        SUMOConnectionError: If SUMO is not available
    """
    logger.info("Starting simulation comparison")
    logger.info("=" * 50)

    logger.info("Running FIXED-TIME simulation...")
    fixed = run_fixed_simulation(green_time=fixed_green_time, gui=gui)

    logger.info("=" * 50)
    logger.info("Running ADAPTIVE (ML) simulation...")
    adaptive = run_adaptive_simulation(rain=rain, gui=gui)

    # Compute improvements (positive = adaptive is better)
    wait_improvement = (
        (fixed["avg_waiting_time"] - adaptive["avg_waiting_time"])
        / max(fixed["avg_waiting_time"], 0.01)
    ) * 100

    queue_improvement = (
        (fixed["avg_queue_length"] - adaptive["avg_queue_length"])
        / max(fixed["avg_queue_length"], 0.01)
    ) * 100

    throughput_diff = adaptive["total_arrived"] - fixed["total_arrived"]

    logger.info("=" * 50)
    logger.info("COMPARISON RESULTS:")
    logger.info(
        f"  Avg Waiting Time: Fixed={fixed['avg_waiting_time']}s, "
        f"Adaptive={adaptive['avg_waiting_time']}s"
    )
    logger.info(
        f"  Avg Queue Length: Fixed={fixed['avg_queue_length']}, "
        f"Adaptive={adaptive['avg_queue_length']}"
    )
    logger.info(
        f"  Throughput: Fixed={fixed['total_arrived']}, "
        f"Adaptive={adaptive['total_arrived']}"
    )
    logger.info(f"  Waiting Time Improvement: {wait_improvement:.1f}%")
    logger.info(f"  Queue Length Improvement: {queue_improvement:.1f}%")

    comparison: Dict[str, Any] = {
        "fixed": fixed,
        "adaptive": adaptive,
        "improvement_wait_pct": round(wait_improvement, 1),
        "improvement_queue_pct": round(queue_improvement, 1),
        "throughput_diff": throughput_diff,
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison saved to: {out_path}")
    return comparison


def print_summary(comparison: Dict[str, Any]) -> None:
    """Print a formatted summary of comparison results."""
    fixed = comparison["fixed"]
    adaptive = comparison["adaptive"]

    print("\n" + "=" * 60)
    print("TRAFFIC SIGNAL OPTIMIZATION - SIMULATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Fixed':<15} {'Adaptive':<15} {'Change':<10}")
    print("-" * 60)
    print(
        f"{'Avg Waiting Time (s)':<25} {fixed['avg_waiting_time']:<15} "
        f"{adaptive['avg_waiting_time']:<15} {comparison['improvement_wait_pct']:+.1f}%"
    )
    print(
        f"{'Avg Queue Length':<25} {fixed['avg_queue_length']:<15} "
        f"{adaptive['avg_queue_length']:<15} {comparison['improvement_queue_pct']:+.1f}%"
    )
    print(
        f"{'Vehicles Completed':<25} {fixed['total_arrived']:<15} "
        f"{adaptive['total_arrived']:<15} {comparison['throughput_diff']:+d}"
    )
    print("=" * 60)

    if comparison["improvement_wait_pct"] > 0:
        print("\nAdaptive ML control OUTPERFORMS fixed-time control!")
    else:
        print("\nFixed-time control performed better in this simulation.")


if __name__ == "__main__":
    import sys

    use_gui = "--gui" in sys.argv
    try:
        result = compare(gui=use_gui)
        print_summary(result)
    except SUMOConnectionError as e:
        print(f"Error: {e}")
        print("Make sure SUMO is installed and available in PATH.")
        sys.exit(1)
