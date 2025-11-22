"""
Enumerative solving algorithm (reusing new prediction service)

Features:
1. Use new algorithm's data loader and prediction service
2. Enumerate all 1/2/3 order path combinations
3. Get all predictions through CacheService
4. Export complete matrix from cache (consistent with old algorithm output)
5. Solve assignment problem with Gurobi
"""

from math import e
import numpy as np
import pickle
import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Manager
from typing import Dict, Tuple

from src.data.loader import DataLoader
from services.predictor import PredictorService
from src.services.cache import CacheService
from src.data.requests import CacheRequest
from src.services.logger import LoggingService


class EnumerativeAlgorithm:
    """Enumerative solving algorithm, functionally equivalent to old algorithm but uses new prediction service."""

    def __init__(self, config: Dict):
        """Initialize algorithm."""
        self.config = config
        self.manager = Manager()
        
        loader = DataLoader(
            path=self.config["data_path"],
            order_seed_end=self.config["order_seed"],
            order_number=self.config["order_number"],
            driver_number=self.config["driver_number"],
            driver_max_numbers=self.config["driver_max_number"],
            driver_seed_end=self.config["driver_seed"],
        )
        self.orders = loader.load_orders()
        self.drivers = loader.load_drivers(self.orders)
        
        self.logging_service = LoggingService(log_file="debug.log", level="INFO", log_dir="logs", manager=self.manager)
        self.logger = self.logging_service.get_logger(__name__)
        
        self.predictor = PredictorService(
            num_workers=self.config["predictor_workers"],
            orders_data=self.orders,
            drivers_data=self.drivers,
            model_paths=self.config["model_path"],
            logging_service=self.logging_service,
        )

        self.cache = CacheService(
            driver_number=len(self.drivers),
            order_number=len(self.orders),
            request_queue=self.manager.Queue(),
            predictor_service=self.predictor,
            manager=self.manager,
            logging_service=self.logging_service,
        )
        self.logging_service.start()

        self.predictor.start()
        self.cache.start()
        
        self.logger.info("EnumerativeAlgorithm initialized successfully")
        
    def load_combinations(self):
        path = self.config["combination_path"]
        order_number = self.config["order_number"]
        with open(path + str(order_number) + "_2.pkl", "rb") as f:
            self.two_combinations = pickle.load(f)
        with open(path + str(order_number) + "_3.pkl", "rb") as f:
            self.three_combinations = pickle.load(f)

    def _generate_path_permutations(self):
        """
        Generate all 1, 2, 3-order path *permutations*
        for populating the ARC cache.
        """
        order_num = len(self.orders)
        self.logger.info(f"Generating permutations for {order_num} orders...")

        # 1-Order paths: [i]
        self.paths_1_order = np.arange(order_num).reshape(-1, 1)

        # 2-Order paths: [i, j] for i != j
        i_indices = np.repeat(np.arange(order_num), order_num - 1)
        j_indices = np.array([j for i in range(order_num) for j in range(order_num) if i != j])
        self.paths_2_order = np.stack([i_indices, j_indices], axis=1)

        # 3-Order paths: [i, j, k]
        # (N * (N-1) * (N-2))
        n = order_num
        paths_3 = []
        for i in range(n):
            for j in range(n):
                if i == j: continue
                for k in range(n):
                    if k == i or k == j: continue
                    paths_3.append([i, j, k])
        self.paths_3_order = np.array(paths_3)
        
        self.logger.info(f"Generated {len(self.paths_1_order)} 1-order, "
                         f"{len(self.paths_2_order)} 2-order, "
                         f"{len(self.paths_3_order)} 3-order permutations.")
        
    def _get_order_batch_mask(self):
        """
        Create order-to-path binary matrix (self.combination_matrix).
        """
        self.logger.info("Building order-to-batch combination matrix...")
        order_numbers = len(self.orders)
        two_combinations = self.two_combinations
        three_combinations = self.three_combinations
        
        C = order_numbers + len(two_combinations) + len(three_combinations)
        combination_matrix = np.zeros((order_numbers, C), dtype=int)

        for i in range(order_numbers):
            combination_matrix[i, i] = 1

        start_two_index = order_numbers
        for idx, (i, j) in enumerate(two_combinations):
            if start_two_index + idx < C:
                combination_matrix[i, start_two_index + idx] = 1
                combination_matrix[j, start_two_index + idx] = 1

        start_three_index = order_numbers + len(two_combinations)
        for idx, (i, j, k) in enumerate(three_combinations):
            if start_three_index + idx < C:
                combination_matrix[i, start_three_index + idx] = 1
                combination_matrix[j, start_three_index + idx] = 1
                combination_matrix[k, start_three_index + idx] = 1
                
        self.combination_matrix = combination_matrix
        self.logger.info("Combination matrix built.")
        
    def _populate_arc_cache(self):
        """Populate all 1, 2, 3 order paths with arc costs in stages."""
        order_num = len(self.orders)
        driver_num = self.config["driver_number"]
        
        self._generate_path_permutations() 

        self.t1_times = np.zeros((driver_num, order_num), dtype=np.float32)
        self.t2_times = np.full((driver_num, order_num, order_num), -1.0, dtype=np.float32)
        
        self.delays_1_arc = np.zeros((driver_num, order_num))
        self.delays_2_arc = np.full((driver_num, order_num, order_num), -1.0)
        self.delays_3_arc = np.full((driver_num, order_num, order_num, order_num), -1.0, dtype=np.float32)

        for driver_idx in range(driver_num):
            self.logger.info(f"Populating ARC cache for Driver {driver_idx}...")

            paths_1_padded = np.pad(self.paths_1_order, ((0, 0), (0, 2)), constant_values=-1)
            
            current_time_1 = np.zeros(len(self.paths_1_order), dtype=np.float32) 
            
            req1 = CacheRequest(driver_idx, paths_1_padded, "arc", current_time_1)
            self.cache.request_queue.put(req1)
            
            delays1, cost_time1 = self.cache.get_response_queue(driver_idx).get()
            
            self.delays_1_arc[driver_idx, :] = delays1
            
            self.t1_times[driver_idx, :] = current_time_1 + cost_time1
            
            paths_2_padded = np.pad(self.paths_2_order, ((0, 0), (0, 1)), constant_values=-1)
            
            idx_i_2 = self.paths_2_order[:, 0]
            current_time_2 = self.t1_times[driver_idx, idx_i_2] 
            
            req2 = CacheRequest(driver_idx, paths_2_padded, "arc", current_time_2)
            self.cache.request_queue.put(req2)
            
            delays2, cost_time2 = self.cache.get_response_queue(driver_idx).get()
            
            idx_j_2 = self.paths_2_order[:, 1]
            
            self.delays_2_arc[driver_idx, idx_i_2, idx_j_2] = delays2
            
            self.t2_times[driver_idx, idx_i_2, idx_j_2] = current_time_2 + cost_time2

            if len(self.paths_3_order) == 0:
                continue 
                
            idx_i_3 = self.paths_3_order[:, 0]
            idx_j_3 = self.paths_3_order[:, 1]
            current_time_3 = self.t2_times[driver_idx, idx_i_3, idx_j_3]
            
            req3 = CacheRequest(driver_idx, self.paths_3_order, "arc", current_time_3)
            self.cache.request_queue.put(req3)

            delays3, incremental_delays_3 = self.cache.get_response_queue(driver_idx).get()
            
            idx_k_3 = self.paths_3_order[:, 2]
            self.delays_3_arc[driver_idx, idx_i_3, idx_j_3, idx_k_3] = delays3
                 
        self.logger.info("ARC cache population complete.")

    def _populate_seq_cache(self):
        """Populate all 2-order and 3-order sequence predictions."""
        self.logger.info("Populating SEQ cache...")
        driver_num = self.config["driver_number"]

        paths_2_comb = np.array(self.two_combinations)
        
        paths_3_comb = np.array(self.three_combinations)

        time_2_comb = np.zeros(len(paths_2_comb), dtype=np.float32)
        time_3_comb = np.zeros(len(paths_3_comb), dtype=np.float32)
        
        for driver_idx in range(driver_num):
            self.logger.predictordebug(f"Driver {driver_idx}: Requesting SEQ predictions...")

            if len(paths_2_comb) > 0:
                req_seq2 = CacheRequest(
                    driver_index=driver_idx, 
                    paths=paths_2_comb, 
                    request_type="seq", 
                    current_time=time_2_comb
                )
                self.cache.request_queue.put(req_seq2)
                _ = self.cache.get_response_queue(driver_idx).get()

            if len(paths_3_comb) > 0:
                req_seq3 = CacheRequest(
                    driver_index=driver_idx, 
                    paths=paths_3_comb, 
                    request_type="seq", 
                    current_time=time_3_comb
                )
                self.cache.request_queue.put(req_seq3)
                _ = self.cache.get_response_queue(driver_idx).get()
        
        self.logger.info("SEQ cache population complete.")
    
    def stop_services(self):
        """Stop all services."""
        self.logger.info("Stopping services...")

        try:
            if self.cache:
                self.logger.info("Stopping cache service...")
                self.cache.stop()
                self.cache = None
        except Exception as e:
            self.logger.error(f"Error stopping cache service: {e}")

        try:
            if self.predictor:
                self.logger.info("Stopping predictor service...")
                self.predictor.stop()
                self.predictor = None
        except Exception as e:
            self.logger.error(f"Error stopping predictor service: {e}")

        self.logger.info("Services stopped")
        
        try:
            import time
            time.sleep(0.1)
            if self.logging_service:
                self.logging_service.stop()
        except Exception as e:
            print(f"Error stopping logging service: {e}")

    def _export_matrices_from_cache(self):
        """Export all prediction results from CacheService and calculate total cost based on optimal sequence."""
        self.logger.info("Exporting matrices from cache...")

        order_num = len(self.orders)
        driver_num = self.config["driver_number"]
        num_two_comb = len(self.two_combinations)
        num_three_comb = len(self.three_combinations)

        self.one_delays = np.zeros((driver_num, order_num))
        self.two_delays = np.zeros((driver_num, num_two_comb))
        self.three_delays = np.zeros((driver_num, num_three_comb))

        total_paths = order_num + num_two_comb + num_three_comb
        self.predict_routes = np.zeros((driver_num, total_paths), dtype=int)

        # 3-order permutation order (must match PredictorService training)
        perm_map_3 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        # 2-order permutation order
        perm_map_2 = [[0, 1], [1, 0]]
        
        two_comb_array = np.array(self.two_combinations)
        three_comb_array = np.array(self.three_combinations)

        for driver_idx in range(driver_num):
            self.logger.predictordebug(f"Exporting data for Driver {driver_idx}...")
            
            self.cache.driver_request_queues[driver_idx].put("EXPORT")
            
            cached_data = self.cache.get_response_queue(driver_idx).get()
            
            if "sequence_result" not in cached_data:
                 self.logger.error(f"Driver {driver_idx} 'EXPORT' response missing 'sequence_result'")
                 continue
            
            sequence_data = cached_data["sequence_result"]

            for o1_idx in range(order_num):
                cost = self.delays_1_arc[driver_idx, o1_idx]
                self.one_delays[driver_idx, o1_idx] = cost
                self.predict_routes[driver_idx, o1_idx] = 0

            for pair_idx, comb in enumerate(two_comb_array):
                o1, o2 = comb[0], comb[1]
                
                best_seq_index = sequence_data[2][o1, o2]
                self.predict_routes[driver_idx, order_num + pair_idx] = best_seq_index
                
                best_perm = perm_map_2[best_seq_index]
                path = (comb[best_perm[0]], comb[best_perm[1]])
                
                cost1 = self.delays_1_arc[driver_idx, path[0]]
                cost2 = self.delays_2_arc[driver_idx, path[0], path[1]]
                total_cost = cost1 + cost2
                
                self.two_delays[driver_idx, pair_idx] = total_cost

            for triple_idx, comb in enumerate(three_comb_array):
                o1, o2, o3 = comb[0], comb[1], comb[2]
                
                best_seq_index = sequence_data[3][o1, o2, o3]
                self.predict_routes[driver_idx, order_num + num_two_comb + triple_idx] = best_seq_index
                
                best_perm = perm_map_3[best_seq_index]
                path = (comb[best_perm[0]], comb[best_perm[1]], comb[best_perm[2]])

                cost1 = self.delays_1_arc[driver_idx, path[0]]
                cost2 = self.delays_2_arc[driver_idx, path[0], path[1]]

                cost3 = self.delays_3_arc[driver_idx, path[0], path[1], path[2]]

                total_cost = cost1 + cost2 + cost3

                self.three_delays[driver_idx, triple_idx] = total_cost

        self.expectation_delay = np.zeros(driver_num)
        for driver_idx in range(driver_num):
            all_delays = np.concatenate([
                self.one_delays[driver_idx], 
                self.two_delays[driver_idx], 
                self.three_delays[driver_idx]
            ])
            self.expectation_delay[driver_idx] = np.mean(all_delays)

        self.logger.info("Matrices exported successfully")

    def _build_cost_matrix(self):
        """Build cost matrix for Gurobi: cost_matrix[driver][path] = cost for driver to execute path."""
        self.logger.info("Building final cost matrix for Gurobi...")

        driver_num = self.config["driver_number"]
        order_num = len(self.orders)
        num_two_comb = len(self.two_combinations)
        num_three_comb = len(self.three_combinations)
        
        total_paths = order_num + num_two_comb + num_three_comb

        cost_matrix = np.zeros((driver_num, total_paths))

        for driver_idx in range(driver_num):
            cost_matrix[driver_idx, :order_num] = self.one_delays[driver_idx]

            start_idx_2 = order_num
            end_idx_2 = start_idx_2 + num_two_comb
            cost_matrix[driver_idx, start_idx_2:end_idx_2] = self.two_delays[driver_idx]

            start_idx_3 = end_idx_2
            cost_matrix[driver_idx, start_idx_3:] = self.three_delays[driver_idx]
            
        self.logger.info("Final cost matrix built.")

    def solve_assignment(self):
        """Solve assignment problem using Gurobi."""
        self.logger.info("Solving assignment problem...")

        driver_num = self.config["driver_number"]
        order_num = len(self.orders)
        
        # Only use the portion of cost matrix needed for the current scenario
        cost_matrix_slice = self.cost_matrix[:driver_num, :]
        rows, cols = cost_matrix_slice.shape

        model = gp.Model("enumerative_assignment")
        model.setParam("OutputFlag", 0)

        # Add binary variables
        x = np.array([[model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for j in range(cols)] for i in range(rows)])

        # Objective function
        x_flatten = x.ravel()
        objective_coeffs = cost_matrix_slice.ravel()
        objective = gp.LinExpr(objective_coeffs, x_flatten)
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraint 1: Each driver selects at most one path
        for i in range(rows):
            model.addConstr(x[i, :].sum() <= 1, f"Driver_{i}_constraint")

        # Constraint 2: Each order must be covered exactly once
        column_sum = x.sum(axis=0)
        
        # (Ensure self.combination_matrix has been created by _get_order_batch_mask)
        combination_exprs = self.combination_matrix @ column_sum
        for i in range(order_num):
            model.addConstr(combination_exprs[i] == 1, f"Order_{i}_constraint")

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            self.logger.info(f"Optimal solution found. Objective: {model.ObjVal:.2f}")

            # Extract solution
            order_batches = []
            
            for i in range(rows):
                for j in range(cols):
                    if x[i, j].X > 0.9:
                        # Determine which type of path this is
                        if j < order_num:
                            # Single order
                            order_batches.append([[j], i])
                        elif j < order_num + len(self.two_combinations):
                            # Two orders
                            pair_idx = j - order_num
                            order_batches.append([list(self.two_combinations[pair_idx]), i])
                        else:
                            # Three orders
                            triple_idx = j - order_num - len(self.two_combinations)
                            order_batches.append([list(self.three_combinations[triple_idx]), i])

            self.assignments = {"routing": order_batches}
            self.logger.info(f"Found {len(order_batches)} assignments")
            
            self.optimal_value = model.ObjVal
        else:
            self.logger.error(f"Optimization failed with status {model.status}")
            self.assignments = None
            self.optimal_value = None

    def output(self) -> Tuple:
        pass


    def run(self):
        """
        Execute the complete enumerative algorithm in sequence.
        This is the main driver that calls all private functions.
        """
        try:
            self.logger.info("--- Enumerative Algorithm Run ---")
            
            # 1. Load combinations needed by Gurobi (e.g., (1, 2), (1, 3), ...)
            self.load_combinations()
            
            # 2. Create Gurobi constraint matrix (must be before solve)
            self._get_order_batch_mask()
            
            # 3. Populate ARC cache (e.g., (1, 2) and (2, 1))
            #    (This function internally includes _generate_path_permutations)
            self._populate_arc_cache()
            
            # 4. Populate SEQ cache (e.g., optimal sequence for {1, 2})
            self._populate_seq_cache()
            
            # 5. Calculate total cost matrix (based on optimal sequences)
            self._export_matrices_from_cache()
            
            # 6. Build cost matrix
            self._build_cost_matrix()
            
            # 7. Solve with Gurobi
            self.solve_assignment()
            
            self.logger.info("--- Run Finished ---")
            return self.assignments, getattr(self, "optimal_value", None)

        except Exception as e:
            self.logger.error(f"Fatal error during algorithm run: {e}", exc_info=True)
            return None
        finally:
            # Ensure services stop on completion or error
            try:
                self.stop_services()
            except Exception as cleanup_error:
                # If logger still works, log error; otherwise use print
                try:
                    self.logger.error(f"Error during cleanup: {cleanup_error}")
                except:
                    print(f"Error during cleanup: {cleanup_error}")
            
            
            
if __name__ == "__main__":
    example_config = {
        "data_path": "data/test_ID/",
        "order_seed": 202,
        "order_number": 45,
        "driver_seed": 202,
        "driver_number": 23,
        "driver_max_number": 23,
        "predictor_workers": 5,
        "model_path": {
            "arc_cost": "data/xgb_model_one_order.pkl",
            "seq_2_order": "data/xgb_routing_model/xgb_model_two_orders.pkl",
            "seq_3_order": "data/xgb_routing_model/xgb_model_three_orders.pkl",
        },
        "combination_path": "data/combinations/",
    }
    
    algorithm = None
    try:
        print("Initializing EnumerativeAlgorithm...")
        algorithm = EnumerativeAlgorithm(example_config)
        
        print("Running algorithm...")
        assignments = algorithm.run()
        
        print("Algorithm completed successfully!")
        print(assignments)
        
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensure algorithm object is properly cleaned up
        if algorithm:
            try:
                print("Performing final cleanup...")
                algorithm.stop_services()
            except Exception as cleanup_e:
                print(f"Error during final cleanup: {cleanup_e}")
        
        print("Program finished.")