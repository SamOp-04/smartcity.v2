"""
SUMO simulation with ML-adaptive traffic signal control.

This module runs traffic simulation using SUMO with green light
timing predicted by the ML model based on real-time vehicle counts.
"""

from typing import Dict, Any, List, Optional
import os
import json

from layer2_ml.predict import predict_green_time
from utils.logging_config import get_logger
from utils.constants import SUMO_SIM_DURATION, SUMO_EDGES, YELLOW_TIME

logger = get_logger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "simulation.sumocfg")


class SUMOConnectionError(Exception):
    """Raised when SUMO connection fails."""
    pass


def count_vehicles_on_edges(
    edge_ids: List[str],
    traci_module: Any = None,
) -> Dict[str, int]:
    """
    Count vehicles by type on given edges (simulating YOLO detection).

    Args:
        edge_ids: List of SUMO edge IDs to count vehicles on
        traci_module: TraCI module (if not provided, imports globally)

    Returns:
        Dict with car_count, bus_truck_count, and bike_count
    """
    if traci_module is None:
        import traci as traci_module

    counts: Dict[str, int] = {"car_count": 0, "bus_truck_count": 0, "bike_count": 0}

    for edge_id in edge_ids:
        try:
            vehicles = traci_module.edge.getLastStepVehicleIDs(edge_id)
        except Exception:
            logger.warning(f"Could not get vehicles on edge: {edge_id}")
            continue

        for vid in vehicles:
            vtype = traci_module.vehicle.getTypeID(vid)
            if vtype == "car":
                counts["car_count"] += 1
            elif vtype == "bus_truck":
                counts["bus_truck_count"] += 1
            elif vtype == "bike":
                counts["bike_count"] += 1

    return counts


def run_adaptive_simulation(
    rain: int = 0,
    sim_duration: int = SUMO_SIM_DURATION,
    gui: bool = False,
    ns_edges: Optional[List[str]] = None,
    ew_edges: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run simulation with ML-predicted signal timing.

    Args:
        rain: Rain condition indicator (0 or 1)
        sim_duration: Total simulation duration in seconds
        gui: Whether to use SUMO GUI
        ns_edges: North-South inbound edge IDs (default from constants)
        ew_edges: East-West inbound edge IDs (default from constants)

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

    # Use configurable edges or defaults
    ns_edges = ns_edges or SUMO_EDGES["NS"]
    ew_edges = ew_edges or SUMO_EDGES["EW"]

    binary = "sumo-gui" if gui else "sumo"
    logger.info(f"Starting adaptive simulation ({sim_duration}s, rain={rain})")

    try:
        traci.start([binary, "-c", CONFIG_PATH, "--no-warnings"])
    except Exception as e:
        raise SUMOConnectionError(f"Failed to start SUMO: {e}") from e

    tls_id = traci.trafficlight.getIDList()[0]

    total_waiting_time: float = 0.0
    step_count: int = 0
    queue_lengths: List[int] = []
    cumulative_arrived: int = 0

    current_phase: int = 0  # 0=NS green, 1=NS yellow, 2=EW green, 3=EW yellow
    phase_timer: int = 0
    yellow_time: int = YELLOW_TIME
    predictions_made: int = 0

    # Initial prediction for NS phase
    counts = count_vehicles_on_edges(ns_edges, traci)
    current_green_time = predict_green_time(
        counts["car_count"],
        counts["bus_truck_count"],
        counts["bike_count"],
        rain,
    )
    predictions_made += 1
    logger.debug(f"Initial NS prediction: {current_green_time}s")

    try:
        while traci.simulation.getTime() < sim_duration:
            traci.simulationStep()
            step_count += 1
            phase_timer += 1

            # Collect metrics
            vehicles = traci.vehicle.getIDList()
            step_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
            total_waiting_time += step_waiting

            queue = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 0.1)
            queue_lengths.append(queue)
            cumulative_arrived += traci.simulation.getArrivedNumber()

            # Phase transitions with ML predictions
            if current_phase == 0 and phase_timer >= current_green_time:
                # NS green -> NS yellow
                traci.trafficlight.setPhase(tls_id, 1)
                current_phase = 1
                phase_timer = 0

            elif current_phase == 1 and phase_timer >= yellow_time:
                # NS yellow -> EW green; predict EW green time
                counts = count_vehicles_on_edges(ew_edges, traci)
                current_green_time = predict_green_time(
                    counts["car_count"],
                    counts["bus_truck_count"],
                    counts["bike_count"],
                    rain,
                )
                predictions_made += 1
                logger.debug(f"EW prediction: {current_green_time}s (counts: {counts})")
                traci.trafficlight.setPhase(tls_id, 2)
                current_phase = 2
                phase_timer = 0

            elif current_phase == 2 and phase_timer >= current_green_time:
                # EW green -> EW yellow
                traci.trafficlight.setPhase(tls_id, 3)
                current_phase = 3
                phase_timer = 0

            elif current_phase == 3 and phase_timer >= yellow_time:
                # EW yellow -> NS green; predict NS green time
                counts = count_vehicles_on_edges(ns_edges, traci)
                current_green_time = predict_green_time(
                    counts["car_count"],
                    counts["bus_truck_count"],
                    counts["bike_count"],
                    rain,
                )
                predictions_made += 1
                logger.debug(f"NS prediction: {current_green_time}s (counts: {counts})")
                traci.trafficlight.setPhase(tls_id, 0)
                current_phase = 0
                phase_timer = 0

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise
    finally:
        traci.close()

    avg_waiting = total_waiting_time / max(step_count, 1)
    avg_queue = sum(queue_lengths) / max(len(queue_lengths), 1)

    results: Dict[str, Any] = {
        "mode": "adaptive",
        "rain": rain,
        "avg_waiting_time": round(avg_waiting, 2),
        "avg_queue_length": round(avg_queue, 2),
        "total_arrived": cumulative_arrived,
        "predictions_made": predictions_made,
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "adaptive_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Adaptive simulation complete: avg_wait={avg_waiting:.1f}s, "
        f"throughput={cumulative_arrived}, predictions={predictions_made}"
    )
    return results


# Type stub for traci module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any


if __name__ == "__main__":
    import sys
    use_gui = "--gui" in sys.argv
    result = run_adaptive_simulation(gui=use_gui)
    print(f"Results: {result}")
