import os
import time
import json
import copy
import pandas as pd
import numpy as np
import traceback

# Import core algorithm classes
try:
    from src.algorithms.enumerative import EnumerativeAlgorithm
    from src.algorithms.orchestrator import Orchestrator
except ImportError:
    print("Error: Cannot import EnumerativeAlgorithm or Orchestrator.")
    print("Please ensure this script is in the same directory as EnumerativeAlgorithm.py and Orchestrator.py.")
    exit(1)

# --- Experiment Configuration ---


SCENARIO_SEEDS_ORDER = [202, 203, 204]
SCENARIO_SEEDS_DRIVER = [202, 203, 204]
SIZE_MAP = {
    # order_num: [driver_num_list]
    30: [10, 12, 15],
    45: [15, 20, 23],
    60: [20, 25, 30],
}

NUM_RUNS = 1

BASE_CONFIG = {
    "data_path": "data/test_ID/",
    "combination_path": "data/combinations/",
    "model_path": {
        "arc_cost": "data/xgb_model_one_order.pkl",
        "seq_2_order": "data/xgb_routing_model/xgb_model_two_orders.pkl",
        "seq_3_order": "data/xgb_routing_model/xgb_model_three_orders.pkl",
    },
    # B&P (Orchestrator) specific defaults
    "predictor_workers": 5,
    "pricing_workers": 15,
    "max_cg_iterations": 20,
    # "k_best" will be dynamically set at runtime
    "N_warmup": 10,
    "N_stagnant": 5,
    "gap_target_for_heuristic": 0.05,
    "gap_terminate": 0.01,
    "rmip_time_limit": 30.0,
    "rmip_mip_gap": 0.01,
    "relative_improvement_threshold": 0.001,
    "max_runtime_seconds": 120,
}

RESULTS_DIR = "results/new_experiment2"

PROGRESS_FILE = os.path.join(RESULTS_DIR, "progress.json")

# --- Helper Functions ---


def save_solution(assignments, filepath):
    """Save assignment dictionary in different formats to JSON file."""
    serializable_assignments = {}
    if not assignments:
        serializable_assignments = None

    elif "routing" in assignments:
        serializable_assignments = assignments

    else:
        for driver_id, details in assignments.items():
            serializable_assignments[str(driver_id)] = {"path": [int(o) for o in details.get("path", [])], "cost": float(details.get("cost", 0.0))}

    try:
        with open(filepath, "w") as f:
            json.dump(serializable_assignments, f, indent=4)
    except Exception as e:
        print(f"  [Error] Failed to save solution to {filepath}: {e}")


def load_progress(progress_file):
    """Load completed scenarios and data from progress.json"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
                completed = set(data.get("completed_scenarios", []))
                summary = data.get("summary_data", [])
                print(f"Loaded progress: found {len(completed)} completed scenarios.")
                return completed, summary
        except Exception as e:
            print(f"Warning: Cannot load {progress_file}, will restart. Error: {e}")
    # If file doesn't exist or fails to load, return empty
    return set(), []


def save_progress(progress_file, completed_scenarios, summary_data):
    """Save current progress to progress.json"""
    try:
        with open(progress_file, "w") as f:
            data = {"completed_scenarios": list(completed_scenarios), "summary_data": summary_data}  # Convert set back to list
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"  [Error] Failed to save progress to {progress_file}: {e}")


# --- Main Run Function ---


def run_experiment():
    print(f"--- Experiment Start (V2) ---")
    print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"Each scenario will run {NUM_RUNS} times to average timing.")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Load data from progress file instead of starting from scratch
    completed_scenarios, summary_data = load_progress(PROGRESS_FILE)

    # Calculate total number of scenarios
    total_scenarios = len(SCENARIO_SEEDS_ORDER) * len(SCENARIO_SEEDS_DRIVER) * sum(len(v) for v in SIZE_MAP.values())
    print(f"Total {total_scenarios} scenarios. Loaded {len(completed_scenarios)} completed scenarios.")

    # Nested loop over seeds
    for order_seed in SCENARIO_SEEDS_ORDER:
        for driver_seed in SCENARIO_SEEDS_DRIVER:
            for order_num, driver_num_list in SIZE_MAP.items():
                for driver_num in driver_num_list:

                    # Update scenario_name
                    scenario_name = f"oseed{order_seed}_dseed{driver_seed}_" f"order{order_num}_driver{driver_num}"
                    scenario_dir = os.path.join(RESULTS_DIR, scenario_name)

                    # Cache/checkpoint check
                    if scenario_name in completed_scenarios:
                        print("\n" + "=" * 80)
                        print(f"►►► {scenario_name} already skipped (found in progress file).")
                        print("=" * 80)
                        continue

                    if not os.path.exists(scenario_dir):
                        os.makedirs(scenario_dir)

                    print("\n" + "=" * 80)
                    print(f"► Starting new scenario: {scenario_name}")
                    print(f"  Config: Order Seed={order_seed}, Driver Seed={driver_seed}, Orders={order_num}, Drivers={driver_num}")
                    print("=" * 80)

                    summary_row = {
                        "scenario": scenario_name,
                        "order_seed": order_seed,
                        "driver_seed": driver_seed,
                        "order_num": order_num,
                        "driver_num": driver_num,
                    }

                    # --- 1. Run Enumerative Algorithm (multiple times) ---
                    print(f"  [1/2] Running: Enumerative Algorithm ({NUM_RUNS} rounds)...")
                    enum_times = []
                    enum_obj = None
                    enum_cols = None

                    for run_idx in range(NUM_RUNS):
                        print(f"    - Round {run_idx+1}/{NUM_RUNS}...")
                        try:
                            config_enum = copy.deepcopy(BASE_CONFIG)
                            config_enum.update(
                                {
                                    "order_seed": order_seed,
                                    "driver_seed": driver_seed,
                                    "order_number": order_num,
                                    "driver_number": driver_num,
                                    "driver_max_number": max(driver_num_list),
                                }
                            )

                            enum_algo = None
                            start_time = time.perf_counter()

                            enum_algo = EnumerativeAlgorithm(config_enum)
                            assignments_enum, obj_val_enum = enum_algo.run()
                            time_enum = time.perf_counter() - start_time

                            enum_times.append(time_enum)

                            # Only save deterministic results on first run
                            if run_idx == 0:
                                enum_obj = obj_val_enum
                                enum_cols = enum_algo.cost_matrix.shape[1]
                                # Save solution
                                save_path = os.path.join(scenario_dir, "enum_solution.json")
                                save_solution(assignments_enum, save_path)

                        except Exception as e:
                            print(f"    - Round {run_idx+1} ✗ Failed: {e}")
                            traceback.print_exc()
                            enum_times.append(np.nan)  # Record failure
                        finally:
                            if "enum_algo" in locals() and enum_algo:
                                try:
                                    enum_algo.stop_services()
                                except Exception as stop_e:
                                    print(f"    [Error] Failed to stop enum_algo services: {stop_e}")

                    # Calculate average time
                    avg_time_enum = np.nanmean(enum_times) if enum_times else None
                    summary_row.update(
                        {
                            "enum_obj": enum_obj,
                            "enum_time_sec": avg_time_enum,
                            "enum_cols": enum_cols,
                        }
                    )
                    print(f"  [1/2] Enumerative Algorithm ✓ Complete (Avg Time: {avg_time_enum:.2f}s, Obj: {enum_obj}, Cols: {enum_cols})")

                    # --- 2. Run B&P Algorithm (multiple times) ---
                    print(f"  [2/2] Running: B&P Algorithm ({NUM_RUNS} rounds)...")
                    bp_times = []
                    bp_obj = None
                    bp_cols = None

                    for run_idx in range(NUM_RUNS):
                        print(f"    - Round {run_idx+1}/{NUM_RUNS}...")
                        try:
                            config_bp = copy.deepcopy(BASE_CONFIG)
                            config_bp.update(
                                {
                                    "order_seed": order_seed,
                                    "driver_seed": driver_seed,
                                    "order_number": order_num,
                                    "driver_number": driver_num,
                                    "driver_max_number": max(driver_num_list),
                                }
                            )

                            # Dynamically set k_best = driver_number
                            config_bp["k_best"] = driver_num
                            print(f"      (Dynamic parameter: k_best set to {driver_num})")

                            bp_algo = None
                            start_time = time.perf_counter()

                            bp_algo = Orchestrator(config_bp)
                            assignments_bp = bp_algo.solve()
                            time_bp = time.perf_counter() - start_time

                            bp_times.append(time_bp)

                            # Only save deterministic results on first run
                            if run_idx == 0:
                                bp_obj = bp_algo.node_manager.global_upper_bound
                                bp_cols = bp_algo.root_node_column_count
                                # Save solution
                                save_path = os.path.join(scenario_dir, "bp_solution.json")
                                save_solution(assignments_bp, save_path)

                        except Exception as e:
                            print(f"    - Round {run_idx+1} ✗ Failed: {e}")
                            traceback.print_exc()
                            bp_times.append(np.nan)  # Record failure
                        finally:
                            if "bp_algo" in locals() and bp_algo:
                                try:
                                    # .solve() internally calls _cleanup()
                                    # but we check just in case
                                    if bp_algo.cache:
                                        bp_algo._cleanup()
                                except Exception as clean_e:
                                    print(f"    [Error] Failed to call bp_algo._cleanup(): {clean_e}")

                    # Calculate average time
                    avg_time_bp = np.nanmean(bp_times) if bp_times else None
                    summary_row.update(
                        {
                            "bp_obj": bp_obj,
                            "bp_time_sec": avg_time_bp,
                            "bp_cols": bp_cols,
                        }
                    )
                    print(f"  [2/2] B&P Algorithm ✓ Complete (Avg Time: {avg_time_bp:.2f}s, Obj: {bp_obj}, Cols: {bp_cols})")

                    # --- 3. Save summary for this scenario ---
                    summary_data.append(summary_row)

                    # Mark this scenario as completed and save progress
                    completed_scenarios.add(scenario_name)
                    save_progress(PROGRESS_FILE, completed_scenarios, summary_data)

                    # Save CSV in real-time (overwrite), always up-to-date
                    df = pd.DataFrame(summary_data)
                    df.to_csv(os.path.join(RESULTS_DIR, "experiment_summary.csv"), index=False)

                    print(f"  --- Scenario {scenario_name} completed and progress saved ---")

    print("\n" + "=" * 80)
    print("★★★ All Experiments Complete ★★★")
    print(f"Summary report saved to: {os.path.join(RESULTS_DIR, 'experiment_summary.csv')}")
    print(f"Progress file saved to: {PROGRESS_FILE}")
    print("=" * 80)

    # Print final DataFrame
    df_final = pd.DataFrame(summary_data)
    print("Final Summary:")
    print(df_final)


# --- Script Entry Point ---

if __name__ == "__main__":
    # (Ensure pandas and numpy are installed: pip install pandas numpy)
    run_experiment()
