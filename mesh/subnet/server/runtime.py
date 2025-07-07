import multiprocessing as mp
import multiprocessing.pool
import threading
from collections import defaultdict
from itertools import chain
from queue import SimpleQueue
from selectors import EVENT_READ, DefaultSelector
from statistics import mean
from time import perf_counter
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
from prefetch_generator import BackgroundGenerator

from mesh.moe.server.module_backend import ModuleBackend
from mesh.subnet.server.task_pool import TaskPoolBase
from mesh.utils import get_logger

logger = get_logger(__name__)


class Runtime(threading.Thread):
    """
    A group of processes that processes incoming requests for multiple module backends on a shared device.
    Runtime is usually created and managed by Server, humans need not apply.

    For debugging, you can start runtime manually with .start() or .run()

    >>> module_backends = {'expert_name': ModuleBackend(**kwargs)}
    >>> runtime = Runtime(module_backends)
    >>> runtime.start()  # start runtime in background thread. To start in current thread, use runtime.run()
    >>> runtime.ready.wait()  # await for runtime to load all experts on device and create request pools
    >>> future = runtime.module_backends['expert_name'].forward_pool.submit_task(*module_inputs)
    >>> print("Returned:", future.result())
    >>> runtime.shutdown()

    :param module_backends: a dict [expert uid -> ModuleBackend]
    :param prefetch_batches: form up to this many batches in advance
    :param sender_threads: dispatches outputs from finished batches using this many asynchronous threads
    :param device: if specified, moves all experts and data to this device via .to(device=device).
      If you want to manually specify devices for each expert (in their forward pass), leave device=None (default)

    :param stats_report_interval: interval to collect and log statistics about runtime performance
    """

    SHUTDOWN_TRIGGER = "RUNTIME SHUTDOWN TRIGGERED"

    def __init__(
        self,
        module_backends: Dict[str, ModuleBackend],
        prefetch_batches=64,
        sender_threads: int = 1,
        device: torch.device = None,
    ):
        super().__init__()
        self.module_backends = module_backends
        self.pools = tuple(chain(*(backend.get_pools() for backend in module_backends.values())))
        self.device, self.prefetch_batches, self.sender_threads = device, prefetch_batches, sender_threads
        self.shutdown_recv, self.shutdown_send = mp.Pipe(duplex=False)
        self.shutdown_trigger = mp.Event()
        self.ready = mp.Event()  # event is set iff server is currently running and ready to accept batches

    def run(self):
        for pool in self.pools:
            if not pool.is_alive():
                pool.start()
        if self.device is not None:
            for backend in self.module_backends.values():
                backend.module.to(self.device)

        with mp.pool.ThreadPool(self.sender_threads) as output_sender_pool:
            try:
                self.ready.set()
                logger.info("Started")

                batch_iterator = self.iterate_minibatches_from_pools()
                if self.prefetch_batches > 0:
                    batch_iterator = BackgroundGenerator(batch_iterator, self.prefetch_batches)

                for pool, batch_index, batch in batch_iterator:
                    logger.debug(f"Processing batch {batch_index} from pool {pool.name}")
                    start = perf_counter()
                    try:
                        outputs, batch_size = self.process_batch(pool, batch_index, *batch)
                        batch_processing_time = perf_counter() - start
                        logger.debug(f"Pool {pool.name}: batch {batch_index} processed, size {batch_size}")

                        output_sender_pool.apply_async(pool.send_outputs_from_runtime, args=[batch_index, outputs])
                    except KeyboardInterrupt:
                        raise
                    except BaseException as exception:
                        logger.exception(f"Caught {exception}, attempting to recover")
                        output_sender_pool.apply_async(pool.send_exception_from_runtime, args=[batch_index, exception])

            finally:
                if not self.shutdown_trigger.is_set():
                    self.shutdown()

    def process_batch(self, pool: TaskPoolBase, batch_index: int, *batch: torch.Tensor) -> Tuple[Any, int]:
        """process one batch of tasks from a given pool, return a batch of results and total batch size"""
        outputs = pool.process_func(*batch)
        return outputs, outputs[0].size(0)

    def shutdown(self):
        """Gracefully terminate a running runtime."""
        logger.info("Shutting down")
        self.ready.clear()

        logger.debug("Terminating pools")
        for pool in self.pools:
            if pool.is_alive():
                pool.terminate()
                pool.join()
        logger.debug("Pools terminated")

        # trigger background thread to shutdown
        self.shutdown_send.send(self.SHUTDOWN_TRIGGER)
        self.shutdown_trigger.set()

    def iterate_minibatches_from_pools(self, timeout=None):
        """Iteratively select non-empty pool with highest priority and loads a batch from that pool"""
        with DefaultSelector() as selector:
            for pool in self.pools:
                selector.register(pool.batch_receiver, EVENT_READ, pool)
            selector.register(self.shutdown_recv, EVENT_READ, self.SHUTDOWN_TRIGGER)

            while True:
                # wait until at least one batch_receiver becomes available
                logger.debug("Waiting for inputs from task pools")
                ready_fds = selector.select()
                ready_objects = {key.data for (key, events) in ready_fds}
                if self.SHUTDOWN_TRIGGER in ready_objects:
                    break  # someone asked us to shutdown, break from the loop

                logger.debug("Choosing the pool with first priority")
                pool = min(ready_objects, key=lambda pool: pool.priority)

                logger.debug(f"Loading batch from {pool.name}")
                batch_index, batch_tensors = pool.load_batch_to_runtime(timeout, self.device)
                logger.debug(f"Loaded batch from {pool.name}")
                yield pool, batch_index, batch_tensors
