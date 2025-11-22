from typing import Optional, Tuple, Dict, List
import numpy as np
import uuid


class CacheRequest:
    """
    The initial request sent TO the CacheService.
    It specifies which driver's cache to check and for which paths.
    """

    def __init__(self, driver_index: int, paths: np.ndarray, request_type: str, current_time: np.ndarray):
        self.driver_index = driver_index
        self.paths = paths
        self.request_type = request_type
        self.current_time = current_time  # Added: This is crucial for prediction features


class PredictionRequest:
    """
    A request object sent FROM CacheService TO PredictorService during a cache miss.
    It contains all necessary information for both communication routing and model prediction.
    """

    def __init__(self, request_type: str, driver_index: int, paths: np.ndarray, current_time: np.ndarray, reply_to: str):

        self.request_type = request_type  # "arc" or "seq"
        self.driver_index = driver_index  # The index of the driver
        self.paths = paths  # The paths/sequences needing prediction
        self.current_time = current_time  # Current time for feature engineering

        # --- Communication routing fields ---
        self.reply_to = reply_to  # The client_id of the response queue
        self.task_id = str(uuid.uuid4())  # A unique ID to match responses with requests

    @property
    def path_len(self) -> int:
        """Dynamically calculates the path length."""
        if self.paths.ndim == 2:
            return self.paths.shape[1]
        return self.paths.shape[0] if self.paths.ndim == 1 else 0


class PricingRequest:
    """
    Pricing subproblem request sent to PricingService
    """

    def __init__(
        self,
        driver_index: int,
        duals: Tuple[np.ndarray, np.ndarray, Dict],  # (lambda, mu, gamma)
        constraints: List[Dict],
        k_best: Optional[int],
        reply_to: str,  # client_id
    ):
        self.driver_index = driver_index
        self.duals = duals
        self.constraints = constraints
        self.k_best = k_best
        self.reply_to = reply_to
        self.task_id = str(uuid.uuid4())


class PricingResponse:
    """
    Solution result returned by PricingService
    """

    def __init__(self, task_id: str, solutions: List[Dict]):  # Keep compatible with existing format
        self.task_id = task_id
        self.solutions = solutions
        # solutions format: [{
        #     "driver_index": int,
        #     "path": np.ndarray,
        #     "reduced_cost": float,
        #     "true_cost": float
        # }, ...]
