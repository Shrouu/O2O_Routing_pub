import heapq
import itertools
import time
import json
from typing import Optional
from src.data.structures import Node
from src.services.logger import LoggingService


class NodeTreeManager:
    """
    Manages the search tree for the Branch-and-Price algorithm.

    Responsible for storing pending nodes, tracking global optimal solutions,
    and implementing node selection and pruning strategies.
    """

    def __init__(self, logging_service: LoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__)
        # Use priority queue (min-heap) to store pending nodes.
        self._node_queue = []

        # For generating unique node IDs
        self._node_counter = itertools.count()

        # Global Upper Bound (for minimization problem), i.e., objective value of best integer solution found so far
        self.global_upper_bound = float("inf")

        # Store detailed records of all nodes (including processed and pruned ones)
        self._node_records = {}

        # Store current best solution (e.g., dictionary of variable values)
        self.best_solution = None
        self.logger.nodetreedebug("NodeTreeManager initialized.")

    def add_node(self, node: Node):
        """
        Adds a new node to the tree and performs pruning check.

        Args:
            node (Node): The new node object to add.
        """
        # This method is now a simple wrapper for add_node_with_metadata
        # and to ensure all nodes are tracked, we force using add_node_with_metadata
        if node.node_id is None:
            node.node_id = next(self._node_counter)

        self.logger.warning(f"Direct call to add_node() is deprecated. " f"Use add_node_with_metadata() to ensure tree is tracked.", stack_info=True)

        # Even if called directly, try to create a minimal record
        if node.node_id not in self._node_records:
            self._node_records[node.node_id] = {
                "node_id": node.node_id,
                "parent_id": None,  # Unknown
                "depth": node.depth,
                "lower_bound": node.lower_bound,
                "status": "pending",
                "branch_arc": None,
                "branch_value": None,
                "arc_flow_at_branch": None,
                "num_fractional_vars": None,
                "gap": None,
                "cg_iterations": None,
                "created_at": time.time(),
                "processed_at": None,
            }

        if node.lower_bound < self.global_upper_bound:
            heapq.heappush(self._node_queue, (node.lower_bound, node.node_id, node))
        else:
            self.logger.nodetreedebug(f"Node {node.node_id} with LB={node.lower_bound:.2f} pruned by bound (GUB={self.global_upper_bound:.2f}).")
            self._node_records[node.node_id]["status"] = "pruned"
            self._node_records[node.node_id]["processed_at"] = time.time()

    # --- New methods ---

    def add_node_with_metadata(
        self, node: Node, parent_id: Optional[int], branch_arc: Optional[tuple], branch_value: Optional[int], arc_flow: Optional[float]
    ):
        """
        Adds a new node to the tree and records its metadata.
        """
        # 1. Assign unique ID
        node_id = next(self._node_counter)
        node.node_id = node_id

        # 2. Calculate Gap
        if self.global_upper_bound != float("inf") and abs(self.global_upper_bound) > 1e-9:
            gap = (node.lower_bound - self.global_upper_bound) / abs(self.global_upper_bound)
        else:
            gap = None

        # 3. Create node record
        record = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": node.depth,
            "lower_bound": node.lower_bound,
            "status": "pending",  # Initial status
            "branch_arc": branch_arc,
            "branch_value": branch_value,
            "arc_flow_at_branch": arc_flow,
            "num_fractional_vars": None,
            "gap": gap,
            "cg_iterations": None,
            "created_at": time.time(),
            "processed_at": None,
        }

        # 4. Execute original add_node logic (pruning check and enqueue)
        if node.lower_bound < self.global_upper_bound:
            heapq.heappush(self._node_queue, (node.lower_bound, node.node_id, node))
            self.logger.nodetreedebug(f"Added Node {node_id} (Parent: {parent_id}, LB: {node.lower_bound:.2f}) to queue.")
        else:
            self.logger.nodetreedebug(
                f"Node {node_id} (Parent: {parent_id}, LB: {node.lower_bound:.2f}) pruned immediately by bound (GUB={self.global_upper_bound:.2f})."
            )
            record["status"] = "pruned"
            record["processed_at"] = time.time()

        # 5. Store record
        self._node_records[node_id] = record

    def update_node_status(
        self,
        node_id: int,
        status: str,
        num_fractional_vars: Optional[int] = None,
        cg_iterations: Optional[int] = None,
        new_lower_bound: Optional[float] = None,
    ):
        """
        Updates the status and solution quality information of a processed node.
        """
        if node_id not in self._node_records:
            self.logger.error(f"Cannot update status for non-existent Node ID: {node_id}")
            return

        record = self._node_records[node_id]
        record["status"] = status
        record["processed_at"] = time.time()

        if num_fractional_vars is not None:
            record["num_fractional_vars"] = num_fractional_vars

        if cg_iterations is not None:
            record["cg_iterations"] = cg_iterations

        if new_lower_bound is not None:
            record["lower_bound"] = new_lower_bound

        # Recalculate Gap (since GUB or LB might have updated)
        if self.global_upper_bound != float("inf") and abs(self.global_upper_bound) > 1e-9:
            record["gap"] = (record["lower_bound"] - self.global_upper_bound) / abs(self.global_upper_bound)
        else:
            record["gap"] = None

        self.logger.nodetreedebug(f"Updated Node {node_id} status to: {status}")

    def export_tree_to_json(self, filepath: str):
        """
        Exports _node_records to a JSON file.
        """
        self.logger.maindebug(f"Exporting branch tree to {filepath}...")

        # Prepare serializable data (handle tuples)
        serializable_records = {}
        for node_id, record in self._node_records.items():
            record_copy = record.copy()
            if record_copy.get("branch_arc") is not None:
                # Convert tuple (i, j) to list [i, j]
                record_copy["branch_arc"] = list(record_copy["branch_arc"])
            serializable_records[node_id] = record_copy

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_records, f, indent=4)
            self.logger.maindebug(f"Successfully exported tree data.")
        except Exception as e:
            self.logger.error(f"Failed to export node tree JSON. Error: {e}")

    # --- Remaining methods ---

    def get_next_node(self) -> Optional[Node]:
        """
        Retrieves the next best node from the queue for processing (Best-First Strategy).

        Returns:
            Optional[Node]: The node with the smallest lower bound in the queue, or None if queue is empty.
        """
        if not self.has_nodes():
            return None

        # heapq.heappop pops and returns the item with the smallest lower_bound
        _, _, next_node = heapq.heappop(self._node_queue)
        self.logger.nodetreedebug(f"Processing next Node {next_node.node_id} (LB={next_node.lower_bound:.2f})")
        return next_node

    def has_nodes(self) -> bool:
        """
        Checks if there are pending nodes.

        Returns:
            bool: True if there are nodes in the queue, False otherwise.
        """
        return len(self._node_queue) > 0

    def update_incumbent(self, solution: dict, objective_value: float) -> bool:
        """
        Updates the global upper bound and best solution when a new, better integer solution is found.

        Args:
            solution (dict): Details of the integer solution.
            objective_value (float): Objective function value of the integer solution.

        Returns:
            bool: True if Upper Bound (UB) was successfully updated (i.e., found a better solution),
                False if this solution is not better than current UB.
        """
        if objective_value < self.global_upper_bound:
            self.logger.nodetreedebug(
                f"New incumbent solution found! Upper bound updated from {self.global_upper_bound:.2f} to {objective_value:.2f}"
            )
            self.global_upper_bound = objective_value
            self.best_solution = solution

            # Update gap for all existing records
            for node_id in self._node_records:
                #
                self.update_node_status(node_id=node_id, status=self._node_records[node_id]["status"])

            self._prune_queue()

            return True  # Return True indicating successful update

        return False  # Return False indicating no update

    def _prune_queue(self):
        """
        Cleans up all unqualified nodes in the queue using the updated global upper bound.
        """
        # Efficient pruning method creates a new, filtered heap
        surviving_nodes = []
        pruned_count = 0

        for item in self._node_queue:
            lb, node_id, node = item
            if lb < self.global_upper_bound:
                surviving_nodes.append(item)
            else:
                # [New] Since it is pruned here, update its status
                self.update_node_status(node_id=node_id, status="pruned")
                pruned_count += 1

        heapq.heapify(surviving_nodes)

        if pruned_count > 0:
            self.logger.nodetreedebug(f"Pruned {pruned_count} nodes from the queue with the new upper bound.")
        self._node_queue = surviving_nodes
        
    def get_global_lower_bound(self) -> float:
        """
        Gets the global lower bound (Best Bound) of the current B&P tree.
        This is the minimum lower bound among all pending nodes.
        """
        # _node_queue is a min-heap storing (lower_bound, node_id, node)
        if not self._node_queue:
            # If queue is empty, search is complete.
            # In this case, if a solution was found, LB should equal UB.
            if self.global_upper_bound != float('inf'):
                return self.global_upper_bound
            else:
                # If queue is empty and no solution found, return -inf
                return float('-inf')

        #
        # Global lower bound is the lower_bound of the heap top element (0th element of tuple)
        global_lb = self._node_queue[0][0]
        return global_lb
