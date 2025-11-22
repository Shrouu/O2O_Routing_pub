from multiprocessing import Queue as MpQueue
from queue import Queue
from threading import Thread
from typing import Dict
import numpy as np

from src.services.logger import LoggingService
from src.data.requests import CacheRequest, PredictionRequest
from services.predictor import PredictorService


class _cache:
    """Internal cache worker for a single driver."""
    
    def __init__(
        self,
        driver_index: int,
        order_number: int,
        client_id: str,
        predictor_request_q: MpQueue,
        predictor_response_q: MpQueue,
        logging_service: LoggingService,
    ):
        self.driver_index = driver_index
        self.order_number = order_number
        self.logger = logging_service.get_logger(__name__)

        self.client_id = client_id
        self.predictor_request_q = predictor_request_q
        self.predictor_response_q = predictor_response_q

        self.arc_calculated: Dict[int, np.ndarray] = {}
        self.seq_calculated: Dict[int, np.ndarray] = {}
        self.arc_delay: Dict[int, np.ndarray] = {}
        self.current_time: Dict[int, np.ndarray] = {}
        self.sequence_result: Dict[int, np.ndarray] = {}

        self.setup_cache_matrix()

        self.logger.cachedebug(f"[Cache Worker {self.driver_index}] Initialized cache for {self.order_number} orders.")

    def setup_cache_matrix(self):
        max_arc_len = 3
        max_seq_len = 3
        for length in range(1, max_arc_len + 1):
            shape = (self.order_number,) * length
            self.arc_calculated[length] = np.zeros(shape, dtype=bool)
            self.arc_delay[length] = np.full(shape, -1.0, dtype=np.float32)
            self.current_time[length] = np.full(shape, -1.0, dtype=np.float32)
        for length in range(2, max_seq_len + 1):
            shape = (self.order_number,) * length
            self.seq_calculated[length] = np.zeros(shape, dtype=bool)
            self.sequence_result[length] = np.full(shape, -1, dtype=np.int8)

    def predict_update(self, missed_paths: np.ndarray, missed_indices_tuple: tuple, path_len: int, request_type: str, current_time: np.ndarray):
        """Handle cache misses by requesting predictions and updating cache."""
        num_misses = missed_paths.shape[0]
        if num_misses == 0:
            return

        if current_time.shape[0] == 1 and num_misses > 1:
            request_times = np.repeat(current_time, num_misses)
        else:
            request_times = current_time

        pred_request = PredictionRequest(
            request_type=request_type, driver_index=self.driver_index, paths=missed_paths, current_time=request_times, reply_to=self.client_id
        )

        self.predictor_request_q.put(pred_request)
        response = self.predictor_response_q.get()

        prediction_result = response["result"]

        if request_type == "arc":
            new_times, new_delays = prediction_result
            self.arc_delay[path_len][missed_indices_tuple] = new_delays
            self.current_time[path_len][missed_indices_tuple] = new_times
            self.arc_calculated[path_len][missed_indices_tuple] = True
        else:
            new_perms = prediction_result
            self.sequence_result[path_len][missed_indices_tuple] = new_perms
            self.seq_calculated[path_len][missed_indices_tuple] = True

    def get_paths_results(self, paths: np.ndarray, request_type: str, current_time: np.ndarray):
        """Check cache, trigger predictions for misses, and return results."""
        if paths.size == 0:
            if request_type == "arc":
                return (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
            else:
                return np.array([], dtype=np.int8)
        
        path_len = np.sum(paths[0] != -1)
        path_indices = tuple(paths[:, :path_len].T)

        calculated_map = self.arc_calculated if request_type == "arc" else self.seq_calculated
        is_calculated = calculated_map[path_len][path_indices]
        miss_mask = ~is_calculated

        if np.any(miss_mask):
            missed_paths_padded = paths[miss_mask]
            missed_indices_tuple = tuple(missed_paths_padded[:, :path_len].T)
            cleaned_missed_paths = missed_paths_padded[:, :path_len]

            num_paths = paths.shape[0]
            if current_time.shape[0] == 1 and num_paths > 1:
                expanded_current_time = np.repeat(current_time, num_paths)
                missed_current_time = expanded_current_time[miss_mask]
            else:
                missed_current_time = current_time[miss_mask]

            # Pass cleaned paths to predict_update
            self.predict_update(cleaned_missed_paths, missed_indices_tuple, path_len, request_type, missed_current_time)

        if request_type == "arc":
            results_delay = self.arc_delay[path_len][path_indices]
            results_time = self.current_time[path_len][path_indices]
            return (results_delay, results_time)
        else:
            return self.sequence_result[path_len][path_indices]

    def run(self, request_queue: Queue, response_queue: Queue):
        """Main loop for cache worker thread."""
        while True:
            try:
                task = request_queue.get()
                if task == "STOP":
                    break

                if task == "EXPORT":
                    response_queue.put({"arc_delay": self.arc_delay, "sequence_result": self.sequence_result})
                    continue

                paths, request_type, current_time = task
                results = self.get_paths_results(paths, request_type, current_time)
                response_queue.put(results)

            except Exception as e:
                self.logger.error(f"[ERROR in Driver {self.driver_index}]: {e}")
                break


class CacheService:
    """Multi-threaded caching service for prediction results."""
    
    def __init__(
        self,
        driver_number: int,
        order_number: int,
        request_queue: MpQueue,
        predictor_service: PredictorService,
        manager: any,
        logging_service: LoggingService,
    ):
        self.driver_number = driver_number
        self.order_number = order_number
        self.request_queue = request_queue
        self.predictor_service = predictor_service
        self.manager = manager
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__)

        self.driver_request_queues: Dict[int, Queue] = {i: Queue() for i in range(driver_number)}

        self.driver_response_queues: Dict[int, MpQueue] = {i: self.manager.Queue() for i in range(driver_number)}

        self.worker_threads: Dict[int, Thread] = {}
        self.dispatcher_thread: Thread = None

        self.logger.cachedebug(f"[CacheService] Initialized with {driver_number} drivers and {order_number} orders.")

    def start(self):
        self.logger.cachedebug("[CacheService] Starting worker processes...")
        for driver_id in range(self.driver_number):
            client_id = f"cache_driver_{driver_id}"

            pred_req_q, pred_res_q = self.predictor_service.get_client_interface(client_id)

            cache_instance = _cache(
                driver_index=driver_id,
                order_number=self.order_number,
                client_id=client_id,
                predictor_request_q=pred_req_q,
                predictor_response_q=pred_res_q,
                logging_service=self.logging_service,
            )

            req_q = self.driver_request_queues[driver_id]
            res_q = self.driver_response_queues[driver_id]

            threads = Thread(target=cache_instance.run, args=(req_q, res_q))
            threads.daemon = True
            self.worker_threads[driver_id] = threads
            threads.start()

        self.logger.cachedebug(f"[CacheService] All {self.driver_number} worker threads started.")

        self.dispatcher_thread = Thread(target=self._run_dispatcher, args=(self.request_queue, self.driver_request_queues, self.logging_service))
        self.dispatcher_thread.daemon = True
        self.dispatcher_thread.start()
        self.logger.cachedebug("[CacheService] Dispatcher thread started.")

    def get_response_queue(self, driver_id: int) -> MpQueue:
        return self.driver_response_queues[driver_id]

    def stop(self):
        self.logger.cachedebug("[CacheService] Stopping all worker processes...")
        self.request_queue.put("STOP")
        for driver_id in range(self.driver_number):
            self.driver_request_queues[driver_id].put("STOP")

        for threads in self.worker_threads.values():
            threads.join()
        self.dispatcher_thread.join()
        self.logger.cachedebug("[CacheService] All threads stopped.")

    @staticmethod
    def _run_dispatcher(request_queue: Queue, driver_request_queues: Dict[int, Queue], logging_service: LoggingService):
        """Dispatch incoming cache requests to appropriate driver worker queues."""
        logger = logging_service.get_logger(__name__)
        while True:
            try:
                request: CacheRequest = request_queue.get()
                if request == "STOP":
                    break
                task = (request.paths, request.request_type, request.current_time)
                driver_request_queues[request.driver_index].put(task)

            except Exception as e:
                logger.error(f"[ERROR in Dispatcher]: {e}")
