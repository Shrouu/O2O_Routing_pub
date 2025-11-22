import numpy as np
import pickle

# ---------------------------------------------------------------------------- #
#                                 Configuration Area                           #
# ---------------------------------------------------------------------------- #
# Please specify the scenario parameters you want to analyze here
TARGET_SCENARIO = {
    # Corresponds to order_numbers in main.py
    "order_number": 45,
    "order_seed": 202,
    # Corresponds to driver_numbers in main.py
    "driver_number": 15,
    # Corresponds to seeds in main.py (this is the base seed)
    "driver_seed": 202,
    # Maximum number of drivers related to driver_numbers list in main.py
    "max_driver_number": 23,
}

# Specify the model to analyze: "benchmark", "driver_feature", or "routing"
MODEL_TO_ANALYZE = "routing"

# --- Path Configuration ---
# Path of the script file relative to the project root directory, used to locate other files
# Assuming your directory structure is:
# Project/
# |- Results/
# |- Resources/
# |- scripts/ (You will place this .py file here)
BASE_PATH = "../"

# ---------------------------------------------------------------------------- #

def find_scenario_index(target_scenario, id_list_path):
    """Find the corresponding index from the ID list based on the target scenario parameters."""
    print("Step 1: Finding result index based on scenario parameters...")
    
    # Copy seed generation logic from main.py to ensure consistency
    order_seed = (
        target_scenario["order_seed"]
        + target_scenario["order_number"] * 100000
        + target_scenario["max_driver_number"] * 1000
    )
    driver_seed = (
        target_scenario["driver_seed"]
        + target_scenario["order_number"] * 100000
        + target_scenario["max_driver_number"] * 1000
    )

    # Build target ID tuple for matching
    target_id = (
        target_scenario["order_number"],
        target_scenario["driver_number"],
        order_seed,
        driver_seed,
    )

    try:
        with open(id_list_path, "rb") as f:
            id_list = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: ID list file not found: {id_list_path}")
        print("Please ensure main.py has run successfully and generated the Results/ID_list.pkl file.")
        return None

    try:
        index = id_list.index(target_id)
        print(f"Success! The index for scenario {target_id} is: {index}\n")
        return index
    except ValueError:
        print(f"Error: Could not find a match for scenario {target_id} in the ID list.")
        return None


def analyze_and_print_solution(target_scenario, model_name, base_path=""):
    """Main function: Load data, reconstruct paths, calculate costs, and format output."""
    
    # --- 1. Locate scenario index ---
    id_list_path = f"{base_path}Results/ID_list.pkl"
    scenario_index = find_scenario_index(target_scenario, id_list_path)

    if scenario_index is None:
        return

    # --- 2. Load required data files ---
    print("Step 2: Loading result and resource files...")
    order_num = target_scenario["order_number"]
    
    try:
        # Load cost matrices and paths
        data_path = f"{base_path}Results/cost_matrices_{order_num}.npz"
        data = np.load(data_path)
        # Note: Results are saved by (seed_combination, driver, orders...), assuming we only care about the first seed combination here
        one_matrix = data["one_matrix"][scenario_index // 9]
        two_matrix = data["two_matrix"][scenario_index // 9]
        three_matrix = data["three_matrix"][scenario_index // 9]
        routes_matrix = data["routes"][scenario_index // 9]

        # Load assignment plan
        assignments_path = f"{base_path}Results/assignments.pkl"
        with open(assignments_path, "rb") as f:
            all_assignments = pickle.load(f)
        assignment_to_check = all_assignments[scenario_index][model_name]

        # Load combination map files
        comb_path = f"{base_path}Resources/combinations/"
        with open(f"{comb_path}{order_num}_2.pkl", "rb") as f:
            two_combinations_map = pickle.load(f)
        with open(f"{comb_path}{order_num}_3.pkl", "rb") as f:
            three_combinations_map = pickle.load(f)
        
        print("All files loaded successfully!\n")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Error loading file - {e}")
        print("Please ensure all required .npz and .pkl files exist in the correct paths.")
        return

    # --- 3. Analyze solution and reconstruct paths ---
    print("Step 3: Analyzing solution, reconstructing paths, and calculating costs...")
    
    # Predefined order permutations
    order_permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    solution_details = []
    total_cost = 0

    for route_info in assignment_to_check:
        batch_orders_raw = route_info[0]
        driver_id = route_info[1]
        
        # Sort orders within the batch to look up in the combination map
        batch_orders = tuple(sorted(batch_orders_raw))
        
        cost = 0.0
        ordered_path = []

        if len(batch_orders) == 1:
            order_idx = batch_orders[0]
            cost = one_matrix[driver_id, order_idx]
            ordered_path = list(batch_orders)
        
        elif len(batch_orders) == 2:
            try:
                batch_idx = two_combinations_map.index(batch_orders)
                global_batch_idx = order_num + batch_idx
                
                permutation_idx = int(routes_matrix[driver_id, global_batch_idx])
                
                if permutation_idx == 0:
                    o1, o2 = batch_orders[0], batch_orders[1]
                    ordered_path = [o1, o2]
                    cost = np.sum(two_matrix[driver_id, o1, o2])
                else: # permutation_idx == 1
                    o1, o2 = batch_orders[1], batch_orders[0]
                    ordered_path = [o1, o2]
                    cost = np.sum(two_matrix[driver_id, o1, o2])
            except ValueError:
                print(f"Warning: Could not find batch in 2-order combinations: {batch_orders}")
                continue

        elif len(batch_orders) == 3:
            try:
                batch_idx = three_combinations_map.index(batch_orders)
                global_batch_idx = order_num + len(two_combinations_map) + batch_idx

                permutation_idx = int(routes_matrix[driver_id, global_batch_idx])
                
                # Get path order
                p = order_permutations[permutation_idx]
                o1, o2, o3 = batch_orders[p[0]], batch_orders[p[1]], batch_orders[p[2]]
                ordered_path = [o1, o2, o3]
                
                # Extract cost from three_matrix
                cost = np.sum(three_matrix[driver_id, o1, o2, o3])
            except ValueError:
                print(f"Warning: Could not find batch in 3-order combinations: {batch_orders}")
                continue
        
        total_cost += cost
        solution_details.append({
            "driver": driver_id,
            "path": ordered_path,
            "cost": cost
        })

    # --- 4. Format output ---
    print("Analysis complete!\n")
    print("-" * 50)
    print(f"Solution results for model '{model_name.upper()}':")
    print("-" * 50)
    print(f"Optimal solution found with value: {total_cost}")
    print("Solution details:")
    
    # Sort by driver ID and output
    solution_details.sort(key=lambda x: x['driver'])
    for detail in solution_details:
        path_str = ", ".join(map(str, detail['path']))
        print(f"  - Driver {detail['driver']:<2}: Path=[{path_str}] with Cost={detail['cost']:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    analyze_and_print_solution(TARGET_SCENARIO, MODEL_TO_ANALYZE, BASE_PATH)