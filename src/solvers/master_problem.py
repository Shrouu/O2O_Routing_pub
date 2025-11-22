import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from src.services.logger import LoggingService
import copy
from typing import Optional, Dict, Any


class MasterProblemSolver:
    """
    Responsible for building, solving, and managing the Restricted Master Problem (RLMP) in Branch-and-Price.
    """

    def __init__(self, orders: list, drivers: list, use_artificials: bool = False, big_m: float = 1e6, logging_service: LoggingService = None):
        self.orders = orders
        self.drivers = drivers
        self.routes = []
        self.theta_vars = []
        self.var_to_route = {}

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model("MasterProblem", env=self.env)

        self.order_constraints = {}
        self.driver_constraints = {}

        self.arc_flow_constraints = {}

        self.art_vars = {}
        self.save_counter = 0
        self.use_artificials = use_artificials
        self.big_m = big_m
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__)

        self._setup_model()

    def _setup_model(self):
        for order_id in self.orders:
            lhs = gp.LinExpr()
            self.order_constraints[order_id] = self.model.addConstr(lhs >= 1, name=f"cover_{order_id}")
        for driver_id in self.drivers:
            lhs = gp.LinExpr()
            self.driver_constraints[driver_id] = self.model.addConstr(lhs <= 1, name=f"assign_{driver_id}")
        self.model.setObjective(gp.LinExpr(), GRB.MINIMIZE)
        if self.use_artificials:
            for o in self.orders:
                av = self.model.addVar(obj=self.big_m, vtype=GRB.CONTINUOUS, name=f"art_{o}")
                self.model.chgCoeff(self.order_constraints[o], av, 1.0)
                self.art_vars[o] = av
        self.model.update()

    def add_columns_batch(self, routes_info: list):
        for route_info in routes_info:
            driver_idx = int(route_info["driver_index"])
            served_orders = route_info.get("path")
            if served_orders is None:
                served_orders = route_info["orders"]
            if hasattr(served_orders, "tolist"):
                served_orders = served_orders.tolist()
            served_orders = [int(o) for o in served_orders if int(o) != -1]
            cost = route_info.get("true_cost")
            if cost is None:
                cost = route_info.get("cost")
            if cost is None:
                raise ValueError("Route info must include 'true_cost' or 'cost'.")

            col = gp.Column()
            for order_idx in served_orders:
                if order_idx in self.order_constraints:
                    col.addTerms(1.0, self.order_constraints[order_idx])
            if driver_idx in self.driver_constraints:
                col.addTerms(1.0, self.driver_constraints[driver_idx])

            path_arcs = self._get_arcs_from_path(served_orders)
            for arc, constr in self.arc_flow_constraints.items():
                if arc in path_arcs:
                    col.addTerms(1.0, constr)

            var_name = f"theta_d{driver_idx}_r{len(self.routes)}"
            theta_var = self.model.addVar(obj=cost, vtype=GRB.CONTINUOUS, name=var_name, column=col)

            self.var_to_route[var_name] = {
                "driver_index": driver_idx,
                "orders": served_orders,
                "true_cost": cost,
                "reduced_cost": route_info.get("reduced_cost", None),
            }
            self.routes.append(self.var_to_route[var_name])
            self.theta_vars.append(theta_var)
        self.model.update()

    def _get_arcs_from_path(self, path: list) -> set:
        """
        Extracts the set of all arcs from an order sequence path.
        Assumes -1 represents the depot.
        """
        arcs = set()
        if not path:
            return arcs

        arcs.add((-1, path[0]))

        for i in range(len(path) - 1):
            arcs.add((path[i], path[i + 1]))
        return arcs

    def get_arc_flows(self) -> dict:
        """
        Calculates the total flow x_ij on each arc after LP solution.
        """
        arc_flows = defaultdict(float)
        if self.model.status != GRB.OPTIMAL:
            self.logger.mpsolverdebug("Warning: Model not solved optimally. Arc flows may be inaccurate.")
            return arc_flows

        for var in self.theta_vars:
            if var.X > 1e-6:
                route_info = self.var_to_route[var.VarName]
                path = route_info.get("orders", [])
                path_arcs = self._get_arcs_from_path(path)
                for arc in path_arcs:
                    arc_flows[arc] += var.X
        return dict(arc_flows)

    def add_arc_flow_constraint(self, arc: tuple, value: int):
        """
        Adds a new arc flow constraint for branching, e.g., x_ij = 1.
        """
        if arc in self.arc_flow_constraints:
            self.logger.mpsolverdebug(f"Warning: Arc flow constraint for {arc} already exists.")
            return

        i, j = arc
        expr = gp.LinExpr()

        for var in self.theta_vars:
            route_info = self.var_to_route[var.VarName]
            path = route_info.get("orders", [])
            path_arcs = self._get_arcs_from_path(path)
            if arc in path_arcs:
                expr.addTerms(1.0, var)

        constr_name = f"force_arc_{i}_{j}"
        new_constr = self.model.addConstr(expr == value, name=constr_name)
        self.arc_flow_constraints[arc] = new_constr
        self.model.update()

    def add_branching_constraint(self, var_to_branch_on, value_to_fix_to):
        if value_to_fix_to == 0:
            var_to_branch_on.ub = 0
        elif value_to_fix_to == 1:
            var_to_branch_on.lb = 1
        self.model.update()

    def solve(self) -> float:
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            return self.model.ObjVal
        else:
            self.logger.mpsolverdebug("Master problem could not be solved to optimality.")
            return float("inf")

    def get_duals_array(self):
        """
        Returns dual variables, now including three parts: lambda (orders), mu (drivers), gamma (forced arcs).
        Return format: (lambda_array, mu_array, gamma_dict)
        """
        import numpy as np

        if self.model.status != GRB.OPTIMAL:
            raise Exception("Cannot get duals, model is not solved to optimality.")

        max_o = max(self.orders) if self.orders else -1
        max_d = max(self.drivers) if self.drivers else -1

        lam = np.zeros(max_o + 1, dtype=np.float32)
        mu = np.zeros(max_d + 1, dtype=np.float32)
        gamma = {}  # Store duals for arc constraints in a dict, key is arc tuple

        for oid, c in self.order_constraints.items():
            lam[int(oid)] = c.Pi
        for did, c in self.driver_constraints.items():
            mu[int(did)] = -float(c.Pi)

        for arc, c in self.arc_flow_constraints.items():
            gamma[arc] = c.Pi

        return lam, mu, gamma  # <-- Return value interface changed

    def get_existing_routes_as_set(self):
        route_set = set()
        for route in self.routes:
            driver_idx = int(route["driver_index"])
            served = route.get("orders", route.get("path", []))
            served = tuple(int(o) for o in served if int(o) != -1)
            route_set.add((driver_idx, served))
        return route_set

    def clone(self):
        new_solver = MasterProblemSolver(
            self.orders, self.drivers, use_artificials=self.use_artificials, big_m=self.big_m, logging_service=self.logging_service
        )
        new_solver.model = self.model.copy()
        new_solver.routes = copy.deepcopy(self.routes)
        new_solver.var_to_route = copy.deepcopy(self.var_to_route)
        new_solver.theta_vars = [new_solver.model.getVarByName(v.VarName) for v in self.theta_vars]
        new_solver.order_constraints = {oid: new_solver.model.getConstrByName(c.ConstrName) for oid, c in self.order_constraints.items()}
        new_solver.driver_constraints = {did: new_solver.model.getConstrByName(c.ConstrName) for did, c in self.driver_constraints.items()}

        new_solver.arc_flow_constraints = {arc: new_solver.model.getConstrByName(c.ConstrName) for arc, c in self.arc_flow_constraints.items()}

        if self.use_artificials and self.art_vars:
            new_solver.art_vars = {oid: new_solver.model.getVarByName(v.VarName) for oid, v in self.art_vars.items()}
        return new_solver
    
    def solve_rmip(self, time_limit: float, mip_gap: float) -> tuple[Optional[float], Optional[Dict[str, Any]]]:
        """
        Solves the Restricted Master Integer Problem (RMIP) on the current column pool as a heuristic.

        Args:
            time_limit (float): Time limit for Gurobi solver (seconds).
            mip_gap (float): MIPGap tolerance for Gurobi solver.

        Returns:
            Tuple[Optional[float], Optional[Dict[str, Any]]]:
                - Objective value of integer solution (if found).
                - Detailed information of integer solution (if found).
                - (None, None) (if no feasible solution found).
        """
        self.logger.mpsolverdebug(
            f"Starting heuristic RMIP solve with TimeLimit={time_limit}s, MIPGap={mip_gap}..."
        )

        # 1. Clone current solver to avoid modifying original LP
        try:
            # Use existing clone method
            rmip_solver = self.clone()
        except Exception as e:
            self.logger.error(f"Failed to clone solver for RMIP: {e}")
            return None, None

        try:
            # 2. Set Gurobi parameters
            rmip_solver.model.setParam("TimeLimit", time_limit)
            rmip_solver.model.setParam("MIPGap", mip_gap)
            rmip_solver.model.setParam("OutputFlag", 0) # Silent solve

            # 3. Change all theta variables type to BINARY
            # Iterate self.theta_vars
            for var in rmip_solver.theta_vars:
                var.vtype = GRB.BINARY

            rmip_solver.model.update() # Apply variable type changes

            # 4. Solve RMIP
            rmip_solver.solve() # Calls Gurobi optimize

            # 5. Check if a feasible integer solution was found
            if rmip_solver.model.SolCount > 0:
                obj_val = rmip_solver.model.ObjVal

                # Extract solution details (consistent with Orchestrator logic)
                solution_detail = {}
                # Use var_to_route mapping
                route_map = rmip_solver.var_to_route

                for var in rmip_solver.theta_vars:
                    if var.X > 0.5: # Check if variable is in solution
                        var_name = var.VarName
                        if var_name in route_map:
                            route_details = route_map[var_name]
                            driver_id = route_details["driver_index"]
                            path = route_details["orders"]
                            cost = route_details["true_cost"]
                            solution_detail[driver_id] = {"path": path, "cost": cost}

                self.logger.mpsolverdebug(f"Heuristic RMIP found a feasible solution. ObjVal: {obj_val:.2f}")
                return obj_val, solution_detail
            else:
                self.logger.mpsolverdebug("Heuristic RMIP did not find a feasible solution within the time limit.")
                return None, None

        except gp.GurobiError as ge:
            self.logger.error(f"Gurobi error during RMIP solve: {ge}")
            return None, None
        except Exception as e:
            self.logger.error(f"Unexpected error during RMIP solve: {e}")
            return None, None
        finally:
            # 6. Must clean up the cloned solver
            rmip_solver.cleanup() # Call existing cleanup method

    def cleanup(self):
        if hasattr(self, "model"):
            self.model.dispose()
        if hasattr(self, "env"):
            self.env.dispose()

    def __del__(self):
        self.cleanup()
