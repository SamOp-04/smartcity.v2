"""
SUMO simulation with fixed-time traffic signal control.

This module runs traffic simulation using SUMO with fixed green/yellow
timing as a baseline for comparison with adaptive control.
"""

from typing import Dict, Any, List, Optional
import os
import json

from utils.logging_config import get_logger
from utils.constants import SUMO_SIM_DURATION, YELLOW_TIME

logger = get_logger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "simulation.sumocfg")


class SUMOConnectionError(Exception):
    """Raised when SUMO connection fails."""
    pass


def run_fixed_simulation(
    green_time: int = 42,
    yellow_time: int = YELLOW_TIME,
    sim_duration: int = SUMO_SIM_DURATION,
    gui: bool = False,
) -> Dict[str, Any]:
    """
    Run simulation with fixed-time signal control.

    Args:
        green_time: Fixed green light duration in seconds
        yellow_time: Yellow light duration in seconds
        sim_duration: Total simulation duration in seconds
        gui: Whether to use SUMO GUI

    Returns:
        Dict containing simulation results

    Raises:
        SUMOConnectionError: If SUMO fails to start
    """
    try:
        import traci
    except ImportError as e:
        raise SUMOConnectionError(
            "SUMO/TraCI not installed. Install SUMO and ensure traci is available."
        ) from e

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"SUMO config not found: {CONFIG_PATH}")

    binary = "sumo-gui" if gui else "sumo"
    logger.info(f"Starting fixed-time simulation ({sim_duration}s, green={green_time}s)")

    try:
        traci.start([binary, "-c", CONFIG_PATH, "--no-warnings"])
    except Exception as e:
        raise SUMOConnectionError(f"Failed to start SUMO: {e}") from e

    total_waiting_time: float = 0.0
    step_count: int = 0
    queue_lengths: List[int] = []
    cumulative_arrived: int = 0

    try:
        while traci.simulation.getTime() < sim_duration:
            traci.simulationStep()
            step_count += 1

            vehicles = traci.vehicle.getIDList()
            step_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
            total_waiting_time += step_waiting

            queue = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 0.1)
            queue_lengths.append(queue)

            cumulative_arrived += traci.simulation.getArrivedNumber()

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise
    finally:
        traci.close()

    avg_waiting = total_waiting_time / max(step_count, 1)
    avg_queue = sum(queue_lengths) / max(len(queue_lengths), 1)

    results: Dict[str, Any] = {
        "mode": "fixed",
        "green_time": green_time,
        "avg_waiting_time": round(avg_waiting, 2),
        "avg_queue_length": round(avg_queue, 2),
        "total_arrived": cumulative_arrived,
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "fixed_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Fixed simulation complete: avg_wait={avg_waiting:.1f}s, throughput={cumulative_arrived}")
    return results


if __name__ == "__main__":
    import sys
    use_gui = "--gui" in sys.argv
    result = run_fixed_simulation(gui=use_gui)
    print(f"Results: {result}")
