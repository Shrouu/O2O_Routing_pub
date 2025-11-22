import time
from multiprocessing import Manager
import uuid
import numpy as np
from src.services.cache import CacheService
from src.data.requests import CacheRequest, PricingRequest, PricingResponse
from src.data.loader import DataLoader
from src.data.structures import Node
from services.predictor import PredictorService
from src.algorithms.node_manager import NodeTreeManager
from src.solvers.master_problem import MasterProblemSolver
from src.solvers.pricing import PricingService
from src.services.logger import LoggingService


class Orchestrator:
    """
    Branch-and-Price algorithm orchestrator.
    Manages initialization, main loop execution, parallel subproblems, branching and cleanup.
    """

    def __init__(self, config: dict):
        
        
        self.start_time=time.time()
        
        self.config = config
        self.pricing_client_id = str(uuid.uuid4())

        self.orders = None
        self.drivers = None
        self.predictor = None
        self.node_manager = None
        self.cache = None
        self.manager = Manager()
        self.branch_counter = 0

        self.processed_nodes = 0
        self.stagnant_nodes = 0
        self.last_lb_at_update = 0.0
        self.last_ub_at_update = float("inf")

        self.relative_improvement_threshold = config.get("relative_improvement_threshold", 0.001)

        self.logging_service = LoggingService(log_file="debug.log", level="MPsolverDebug", log_dir="logs", manager=self.manager)
        self.logger = self.logging_service.get_logger(__name__)

    def _setup(self):
        """Load data and create service instances."""
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

        self.node_manager = NodeTreeManager(self.logging_service)

        self.pricing_service = PricingService(
            self.config["pricing_workers"],
            orders_data=self.orders,
            cache_req_q=self.cache.request_queue,
            cache_resp_queues=self.cache.driver_response_queues,
            logging_service=self.logging_service,
        )

        self.logging_service.start()

        self.predictor.start()
        self.pricing_service.start()
        self.cache.start()

        self.logger.maindebug("Setup orchestrator successfully.")

    def _solve_node_with_cg(self, node: Node) -> bool:
        """
        Perform column generation (CG) for a given node.
        """
        self.logger.maindebug(f"\n--- Solving Node {node.node_id} (LB={node.lower_bound:.2f}, Depth={node.depth}) ---")
        available_drivers = set(range(len(self.drivers)))
        available_orders = set(range(len(self.orders)))
        self.logger.maindebug(f"Initial available drivers: {available_drivers}")
        self.logger.maindebug(f"Initial available orders: {available_orders}")

        if node.constraints:
            for const in node.constraints:
                if const.get("type") == "MUST_TAKE_ROUTE":
                    driver_to_assign = const["driver_index"]
                    orders_to_assign = const["route"]

                    if driver_to_assign in available_drivers:
                        available_drivers.remove(driver_to_assign)

                    for order_id in orders_to_assign:
                        available_orders.discard(order_id)
                    self.logger.maindebug(f"must_take constraints applied: removed driver {driver_to_assign} and orders {orders_to_assign}")

        if not available_drivers or not available_orders:
            self.logger.maindebug("Node solved by constraint decomposition.")
            node.solver.solve()
            return True

        try:
            pricing_req_q, pricing_resp_q = self.pricing_service.get_client_interface(self.pricing_client_id)
        except Exception as e:
            self.logger.error(f"Failed to get pricing service interface: {e}")
            return False
        cg_iters = 0
        for i in range(self.config["max_cg_iterations"]):
            cg_iters = i + 1
            self.logger.maindebug(f"  CG Iteration {i+1}...")
            node.solver.solve()

            self.logger.maindebug(f"solved main problem, ObjVal={node.solver.model.ObjVal:.2f}")
            dual_lambda, dual_mu, dual_gamma = node.solver.get_duals_array()
            duals = (dual_lambda, dual_mu, dual_gamma)

            num_tasks_sent = len(available_drivers)
            for driver_index in available_drivers:
                pricing_request = PricingRequest(
                    driver_index=driver_index,
                    duals=duals,
                    constraints=node.constraints,
                    k_best=self.config.get("k_best", None),
                    reply_to=self.pricing_client_id,
                )
                pricing_req_q.put(pricing_request)

            results = []
            for _ in range(num_tasks_sent):
                response: PricingResponse = pricing_resp_q.get()
                results.append(response.solutions)

            print(f"  ColGen Iteration {i+1}: Collected pricing results from {num_tasks_sent} drivers.")

            new_columns_found = []
            for driver_results in results:
                for col in driver_results:
                    if col["reduced_cost"] < -1e-6:
                        new_columns_found.append(col)

            existing_routes_set = node.solver.get_existing_routes_as_set()
            truly_new_columns = []
            for col in new_columns_found:
                orders_in_path = tuple(int(x) for x in col["path"])
                route_tuple = (int(col["driver_index"]), orders_in_path)
                if route_tuple not in existing_routes_set:
                    truly_new_columns.append(col)

            self.logger.maindebug(f"  After filtering duplicates, {len(truly_new_columns)} truly new columns remain.")

            if not truly_new_columns:
                self.logger.maindebug("  Column generation converged for this node.")
                node.lower_bound = node.solver.model.ObjVal
                return True, cg_iters

            start_time = time.time()
            batch_routes_to_add = []
            for col in truly_new_columns:
                path_wo_depot = [int(x) for x in col["path"]]
                batch_routes_to_add.append(
                    {
                        "true_cost": float(col["true_cost"]),
                        "driver_index": int(col["driver_index"]),
                        "path": np.asarray(path_wo_depot, dtype=np.int32),
                    }
                )
            if batch_routes_to_add:
                node.solver.add_columns_batch(batch_routes_to_add)
            end_time = time.time()
            self.logger.maindebug(f"  Batch adding columns completed in {end_time - start_time:.2f} seconds.")
        self.logger.warning("  Warning: Max CG iterations reached.")
        return True, cg_iters

    def _branch_arc(self, parent_node: Node):
        """Execute branching logic."""
        arc_flows = parent_node.solver.get_arc_flows()

        best_arc_to_branch = None
        min_dist_to_half = 0.5
        for arc, flow in arc_flows.items():
            if flow > 1e-6 and flow < 1 - 1e-6:
                dist = abs(flow - 0.5)
                if dist < min_dist_to_half:
                    min_dist_to_half = dist
                    best_arc_to_branch = arc

        if best_arc_to_branch is None:
            return

        self.logger.maindebug(f"    -> Branching on arc {best_arc_to_branch} with flow {arc_flows[best_arc_to_branch]:.2f}")

        self.branch_counter += 1

        i, j = best_arc_to_branch
        for branch_value in [0, 1]:
            self.logger.maindebug(f"  Creating branch node where arc ({i},{j}) = {branch_value}")

            child_solver = parent_node.solver.clone()
            parent_constraints = parent_node.constraints

            new_constraint = {"type": "FORBID_ARC", "arc": (i, j)} if branch_value == 0 else {"type": "FORCE_ARC", "arc": (i, j)}

            child_constraints = parent_constraints + [new_constraint]

            if branch_value == 1:
                child_solver.add_arc_flow_constraint(arc=(i, j), value=1)
            else:
                child_solver.add_arc_flow_constraint(arc=(i, j), value=0)

            new_lb = child_solver.solve()

            if new_lb == float("inf"):
                new_lb = parent_node.lower_bound
                self.logger.maindebug(f"  RMP infeasible for arc ({i},{j})={branch_value}, " f"using parent LB={new_lb:.2f} (will try CG later)")
            else:
                self.logger.maindebug(f"  RMP solved for arc ({i},{j})={branch_value}, LB={new_lb:.2f}")

            child_node = Node(
                lower_bound=new_lb,
                constraints=child_constraints,
                solver=child_solver,
                depth=parent_node.depth + 1,
            )
            # self.node_manager.add_node(child_node)
            self.node_manager.add_node_with_metadata(
                node=child_node,
                parent_id=parent_node.node_id,
                branch_arc=best_arc_to_branch,
                branch_value=branch_value,
                arc_flow=arc_flows[best_arc_to_branch],
            )

    def _init_columns_greedy_constructive(self):
        """
        Initialize columns using greedy constructive heuristic.
        Strategy: Sequentially select a seed point from the "bottom-left corner",
                  match it with the two nearest neighbors, then remove these three
                  points from the pool until all points are assigned.
        """
        self.logger.maindebug("  Initializing root node with greedy constructive heuristic...")

        locations = self.orders.locations
        num_orders = len(locations)
        available_indices = set(range(num_orders))
        batches = []

        while len(available_indices) >= 3:
            current_pool_list = list(available_indices)
            pool_locs = locations[current_pool_list]

            idx_of_seed_in_pool = np.argmin(pool_locs[:, 0])
            seed_index = current_pool_list[idx_of_seed_in_pool]

            neighbor_pool_indices = list(available_indices - {seed_index})
            neighbor_locs = locations[neighbor_pool_indices]

            seed_loc = locations[seed_index]

            distances = np.linalg.norm(neighbor_locs - seed_loc, axis=1)

            indices_of_neighbors_in_pool = np.argsort(distances)[:2]

            neighbor1_index = neighbor_pool_indices[indices_of_neighbors_in_pool[0]]
            neighbor2_index = neighbor_pool_indices[indices_of_neighbors_in_pool[1]]

            new_batch = [seed_index, neighbor1_index, neighbor2_index]
            batches.append(new_batch)

            available_indices.remove(seed_index)
            available_indices.remove(neighbor1_index)
            available_indices.remove(neighbor2_index)

        if available_indices:
            batches.append(list(available_indices))
            self.logger.maindebug(f"  Created {len(batches)-1} batches of 3, and 1 leftover batch of size {len(available_indices)}.")
        else:
            self.logger.maindebug(f"  Created {len(batches)} batches of 3.")

        return batches

    def init_columns(self, root_solver: MasterProblemSolver):
        """Receive batch list, execute calling CacheService, cost calculation and column addition."""
        driver_ids = list(range(len(self.drivers)))
        cache_req_q = self.cache.request_queue
        initial_routes_to_add_batch = []

        batches = self._init_columns_greedy_constructive()

        for batch in batches:
            if not batch:
                continue

            for driver_index in driver_ids:
                cache_resp_q = self.cache.get_response_queue(driver_index)

                final_seq = batch
                if len(batch) in (2, 3):
                    path_to_predict = np.array([batch], dtype=np.int32)
                    seq_request = CacheRequest(
                        driver_index=driver_index, paths=path_to_predict, request_type="seq", current_time=np.array([0.0], dtype=np.float32)
                    )
                    cache_req_q.put(seq_request)
                    best_perms_indices = cache_resp_q.get()
                    best_idx = int(best_perms_indices[0])
                    if len(batch) == 2:
                        perms = [(0, 1), (1, 0)]
                        final_seq = [batch[p] for p in perms[best_idx]]
                    else:
                        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                        final_seq = [batch[p] for p in perms[best_idx]]

                total_delay, current_time_val = 0.0, 0.0
                current_path = []
                for next_order in final_seq:
                    current_path.append(next_order)
                    arc_request = CacheRequest(
                        driver_index=driver_index,
                        paths=np.array([current_path], dtype=np.int32),
                        request_type="arc",
                        current_time=np.array([current_time_val], dtype=np.float32),
                    )
                    cache_req_q.put(arc_request)
                    delays, new_times = cache_resp_q.get()
                    delay_on_this_arc = delays[0]
                    current_time_val += new_times[0]
                    total_delay += max(0.0, delay_on_this_arc)

                initial_routes_to_add_batch.append(
                    {
                        "true_cost": total_delay,
                        "driver_index": int(driver_index),
                        "path": np.asarray(final_seq, dtype=np.int32),
                    }
                )

        for driver_index in driver_ids:
            cache_resp_q = self.cache.get_response_queue(driver_index)

            for order_index in range(len(self.orders)):
                single_order_path = [order_index]

                arc_request = CacheRequest(
                    driver_index=driver_index,
                    paths=np.array([single_order_path], dtype=np.int32),
                    request_type="arc",
                    current_time=np.array([0.0], dtype=np.float32),
                )
                cache_req_q.put(arc_request)
                delays, new_times = cache_resp_q.get()
                single_order_delay = max(0.0, delays[0])

                initial_routes_to_add_batch.append(
                    {
                        "true_cost": single_order_delay,
                        "driver_index": int(driver_index),
                        "path": np.asarray(single_order_path, dtype=np.int32),
                    }
                )

        if initial_routes_to_add_batch:
            root_solver.add_columns_batch(initial_routes_to_add_batch)

        self.logger.maindebug(
            f"  Added {len(initial_routes_to_add_batch)} initial columns to the root node solver (including {len(self.drivers) * len(self.orders)} single-order routes)."
        )

    def _print_performance_summary(self):
        """Print final performance statistics summary."""
        total_runtime = time.time() - self.total_start_time
        self.logger.maindebug("\n--- [Performance Summary] ---")
        self.logger.maindebug(f"Optimal solution found with value: {self.node_manager.global_upper_bound}")
        self.logger.maindebug(f"  - Total Runtime: {total_runtime:.2f} seconds")
        self.logger.maindebug(f"  - Branch Operations: {self.branch_counter}")
        self.logger.maindebug(f"  - Final Root Columns: {self.root_node_column_count}")

        driver_num = self.config["driver_number"]
        order_num = self.config["order_number"]
        theoretical_max = driver_num * (order_num**3)

        self.logger.maindebug("\n  --- Column Generation Analysis ---")
        self.logger.maindebug(f"  - Theoretical Max Columns: {theoretical_max} (based on D * N^3)")

        if theoretical_max > 0:
            gap_percentage = (self.root_node_column_count / theoretical_max) * 100
            self.logger.maindebug(f"  - Actual columns as percentage of theoretical: {gap_percentage:.4f}%")

        self.logger.maindebug("--- [End of Summary] ---")

    def solve(self):
        """Start and execute complete Branch-and-Price algorithm."""
        self.total_start_time = time.time()
        self._setup()

        stop_time=time.time()
        
        print("Setup Time:",stop_time - self.start_time)
        
        self.logger.maindebug("\n--- [Phase 2: Solving Root Node] ---")
        root_solver = MasterProblemSolver(list(range(len(self.orders))), list(range(len(self.drivers))), logging_service=self.logging_service)
        self.init_columns(root_solver)
        root_node = Node(lower_bound=0.0, constraints=[], solver=root_solver, depth=0)

        converged, cg_iters = self._solve_node_with_cg(root_node)
        self.root_node_column_count = len(root_node.solver.routes)
        # self.node_manager.add_node(root_node)
        
        self.last_lb_at_update = root_node.lower_bound

        self.logger.maindebug("\n--- [Phase 2.5: Root Node Heuristic] ---")
        self.logger.maindebug("Solving RMIP on root node columns to get initial Upper Bound...")

        rmip_time_limit = self.config.get("rmip_time_limit", 30.0)
        rmip_mip_gap = self.config.get("rmip_mip_gap", 0.01)

        try:
            obj_val, solution_detail = root_node.solver.solve_rmip(time_limit=rmip_time_limit, mip_gap=rmip_mip_gap)

            if obj_val is not None:
                self.logger.maindebug(f"Root heuristic found initial solution with ObjVal: {obj_val:.2f}")

                was_updated = self.node_manager.update_incumbent(solution_detail, obj_val)

                if was_updated:
                    self.last_ub_at_update = obj_val
            else:
                self.logger.maindebug("Root heuristic did not find an initial solution.")

        except Exception as e:
            self.logger.error(f"Error during root node RMIP heuristic: {e}")

        self.node_manager.add_node_with_metadata(node=root_node, parent_id=None, branch_arc=None, branch_value=None, arc_flow=None)

        self.logger.maindebug("\n--- [Phase 3: Branch and Price Main Loop] ---")
        
        termination_reason = "Natural Convergence (Tree Exhausted)"
        
        while self.node_manager.has_nodes():
            
            max_time = self.config.get("max_runtime_seconds", 3600)
            
            if (time.time() - self.total_start_time) > max_time:
                self.logger.maindebug("\n--- [SAFETY NET TERMINATION] ---")
                self.logger.maindebug(f"  Runtime exceeded limit of {max_time} seconds.")
                termination_reason = "Safety Net (Time Limit Reached)"
                break
            
            current_node = self.node_manager.get_next_node()

            self.processed_nodes += 1

            if current_node.lower_bound >= self.node_manager.global_upper_bound:
                self.logger.maindebug(f"Pruning Node {current_node.node_id} (LB={current_node.lower_bound:.2f}) by bound.")
                status = "infeasible" if current_node.lower_bound == float("inf") else "pruned"
                self.node_manager.update_node_status(node_id=current_node.node_id, status=status)
                continue

            converged, cg_iters = self._solve_node_with_cg(current_node)
            if not converged:
                self.logger.error(f"Failed to solve Node {current_node.node_id}. Marking as error.")
                self.node_manager.update_node_status(node_id=current_node.node_id, status="error", cg_iterations=cg_iters)
                continue

            solution_vars = current_node.solver.theta_vars
            solution_vals = [var.X for var in solution_vars]

            is_integer = all(abs(val - round(val)) < 1e-6 for val in solution_vals)

            num_fractional = sum(1 for val in solution_vals if abs(val - round(val)) >= 1e-6)
            status = "integer" if num_fractional == 0 else "processed"

            self.node_manager.update_node_status(
                node_id=current_node.node_id,
                status=status,
                num_fractional_vars=num_fractional,
                cg_iterations=cg_iters,
                new_lower_bound=current_node.lower_bound,
            )

            ub_improved_this_node = False

            if is_integer:
                self.logger.maindebug(f"!!! Integer solution found at Node {current_node.node_id} with value {current_node.lower_bound:.2f} !!!")

                detailed_solution = {}
                route_map = current_node.solver.var_to_route

                for var in solution_vars:
                    if var.X > 0.5:
                        var_name = var.VarName
                        if var_name in route_map:
                            route_details = route_map[var_name]
                            driver_id = route_details["driver_index"]
                            path = route_details["orders"]
                            cost = route_details["true_cost"]

                            detailed_solution[driver_id] = {"path": path, "cost": cost}

                ub_improved_this_node = self.node_manager.update_incumbent(solution=detailed_solution, objective_value=current_node.lower_bound)
            else:
                self._branch_arc(current_node)
            
            current_global_lb = self.node_manager.get_global_lower_bound()
            lb_improved = False
            if current_global_lb > self.last_lb_at_update + (abs(self.last_lb_at_update) * self.relative_improvement_threshold):
                lb_improved = True
                self.last_lb_at_update = current_global_lb

            # Check if UB has "substantially improved"
            if ub_improved_this_node:
                # Record and reset counter
                self.last_ub_at_update = self.node_manager.global_upper_bound
                self.stagnant_nodes = 0
                self.logger.maindebug(f"  -> UB improved naturally. Resetting stagnant nodes to 0.")
            elif lb_improved:
                # Record and reset counter
                self.stagnant_nodes = 0
                self.logger.maindebug(f"  -> LB improved. Resetting stagnant nodes to 0.")
            else:
                # No improvement, increment counter
                self.stagnant_nodes += 1
                self.logger.maindebug(f"  -> No significant improvement. Stagnant nodes: {self.stagnant_nodes}")


            # --- Step 5.2: Heuristic Trigger ---

            # Get parameters from config
            N_warmup = self.config.get("N_warmup", 20)
            N_stagnant = self.config.get("N_stagnant", 15)
            gap_target_heuristic = self.config.get("gap_target_for_heuristic", 0.05)

            current_global_ub = self.node_manager.global_upper_bound

            gap = float('inf')
            if current_global_ub != float('inf'):
                gap = (current_global_ub - current_global_lb) / max(1.0, abs(current_global_ub))

            if (self.processed_nodes >= N_warmup and
                self.stagnant_nodes >= N_stagnant and
                current_global_ub != float('inf') and
                gap <= gap_target_heuristic):

                self.logger.maindebug(
                    f"--- Triggering RMIP Heuristic (Processed={self.processed_nodes}, Stagnant={self.stagnant_nodes}, Gap={gap*100:.2f}%) ---"
                )

                rmip_time = self.config.get("rmip_time_limit", 30.0)
                rmip_gap = self.config.get("rmip_mip_gap", 0.01)

                try:
                    obj_val, sol_detail = current_node.solver.solve_rmip(rmip_time, rmip_gap)

                    if obj_val is not None:
                        heuristic_updated_ub = self.node_manager.update_incumbent(sol_detail, obj_val)
                        if heuristic_updated_ub:
                            self.logger.maindebug("  -> Heuristic found new UB. Resetting stagnant nodes to 0.")
                            self.last_ub_at_update = self.node_manager.global_upper_bound
                            self.stagnant_nodes = 0
                        else:
                            self.logger.maindebug("  -> Heuristic ran, but did not improve UB.")
                except Exception as e:
                    self.logger.error(f"Error during triggered RMIP heuristic: {e}")
                    
                final_global_lb = self.node_manager.get_global_lower_bound()
                final_global_ub = self.node_manager.global_upper_bound
                gap_terminate = self.config.get("gap_terminate", 0.01)

                if final_global_ub != float('inf'):
                    final_gap = (final_global_ub - final_global_lb) / max(1.0, abs(final_global_ub))

                    if final_gap <= gap_terminate:
                        self.logger.maindebug("\n--- [HEURISTIC TERMINATION] ---")
                        self.logger.maindebug(f"  Final Gap ({final_gap*100:.4f}%) <= Target ({gap_terminate*100:.4f}%)")
                        self.logger.maindebug(f"  Global LB: {final_global_lb:.4f}")
                        self.logger.maindebug(f"  Global UB: {final_global_ub:.4f}")
                        self.logger.maindebug("  Algorithm terminating early based on gap tolerance.")
                        
                        termination_reason = "Heuristic Stop (Gap Tolerance Met)"
                        
                        break


        self.logger.maindebug("\n--- [Phase 4: Algorithm Finished] ---")
        
        self.logger.maindebug(f"  Termination Reason: {termination_reason}")
        
        self.logger.maindebug(f"Optimal solution found with value: {self.node_manager.global_upper_bound}")
        self.logger.maindebug("Solution details:")

        if self.node_manager.best_solution:
            sorted_solution = sorted(self.node_manager.best_solution.items())

            for driver_id, details in sorted_solution:
                path_str = ", ".join(map(str, details["path"]))
                cost = details["cost"]
                self.logger.maindebug(f"  - Driver {driver_id}: Path=[{path_str}] with Cost={cost:.4f}")
        else:
            self.logger.maindebug("  No integer solution found.")

        self._print_performance_summary()
        self._cleanup()
        return self.node_manager.best_solution

    def _cleanup(self):
        """Shutdown all services."""
        self.logger.maindebug("\n--- [Phase 5: Cleanup] ---")

        if self.node_manager:
            try:
                # Ensure log directory exists
                log_dir_path = self.logging_service.log_path
                export_path = log_dir_path / "branch_tree.json"
                self.node_manager.export_tree_to_json(str(export_path))
            except Exception as e:
                self.logger.error(f"Failed to export branch tree during cleanup: {e}")

        if self.cache:
            self.cache.stop()
        if self.predictor:
            self.predictor.stop()
        if self.pricing_service:
            self.pricing_service.stop()

        # Stop logging service last to capture other service shutdown messages
        if self.logging_service:
            self.logging_service.stop()


if __name__ == "__main__":
    # Example configuration
    config = {
        "data_path": "data/test_ID/",
        "model_path": {
            "arc_cost": "data/xgb_model_one_order.pkl",
            "seq_2_order": "data/xgb_routing_model/xgb_model_two_orders.pkl",
            "seq_3_order": "data/xgb_routing_model/xgb_model_three_orders.pkl",
        },
        "order_seed": 202,
        "order_number": 30,
        "driver_seed": 202,
        "driver_number": 15,
        "driver_max_number": 15,
        "predictor_workers": 5,
        "pricing_workers": 15,
        "max_cg_iterations": 20,
        "k_best": 15,
        # --- Heuristic termination strategy configuration ---
        "N_warmup": 10,  # (Trigger condition) Process at least N nodes before considering heuristic
        "N_stagnant": 5,  # (Trigger condition) LB/UB has not "substantially improved" for N consecutive nodes
        "gap_target_for_heuristic": 0.05,  # (Trigger condition) 5% gap, trigger only when gap is below this value
        "gap_terminate": 0.01,  # (Termination condition) 1% gap, terminate when gap is below this value
        "rmip_time_limit": 30.0,  # (Heuristic config) Time limit for each RMIP solve (seconds)
        "rmip_mip_gap": 0.01,  # (Heuristic config) 1% MIPGap for RMIP solve itself
        "relative_improvement_threshold": 0.001,  # (Stagnation threshold) 0.1%, used to judge if LB/UB has "substantially improved"
        "max_runtime_seconds": 120, # (e.g. 2 minutes)
        # ------------------------------------
    }

    # Create and run orchestrator
    orchestrator = None  # Declare outside try block first
    try:
        # Create and run orchestrator
        orchestrator = Orchestrator(config)
        best_solution = orchestrator.solve()

    except Exception as e:
        # Catch any fatal errors in main process
        print(f"\n--- FATAL ERROR IN MAIN PROCESS ---")
        import traceback

        traceback.print_exc()  # Print full error stack
        print(f"-----------------------------------\n")

    finally:
        # Ensure cleanup of child services even if a crash occurs
        if orchestrator:
            print("Attempting to cleanup services...")
            # We cannot call _cleanup() from orchestrator.solve()
            # Manually stop services since logger may also be broken
            try:
                if orchestrator.cache:
                    orchestrator.cache.stop()
                if orchestrator.predictor:
                    orchestrator.predictor.stop()
                if orchestrator.pricing_service:
                    orchestrator.pricing_service.stop()
                print("Cleanup attempt finished.")
            except Exception as ce:
                print(f"Error during cleanup: {ce}")
