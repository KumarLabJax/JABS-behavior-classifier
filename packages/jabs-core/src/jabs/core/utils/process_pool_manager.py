import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import shared_memory
from typing import Any

logger = logging.getLogger(__name__)

MAX_POOL_WORKERS = 6


def _noop() -> None:
    """No-op function for warming up worker processes.

    Must be at module level to be pickleable by ProcessPoolExecutor.
    """
    return None


class ProcessPoolManager:
    """
    Manage a shared ProcessPoolExecutor with warm-up and safe shutdown.

    Attributes:
        _max_workers (int | None): Maximum number of worker processes. Passed to
            ProcessPoolExecutor when created.
        _initializer (Callable[..., object] | None): Optional function executed in
            each worker process when it starts.
        _initargs (tuple[object, ...]): Arguments passed to the initializer.
        _name (str): Logical name for debugging/logging.
        _executor (ProcessPoolExecutor | None): The lazily-created underlying
            process pool. None until first use.
        _lock (threading.RLock): Protects access to `_executor` and `_is_shutdown`.
        _is_shutdown (bool): Whether shutdown() has been called. Prevents reuse once
            the pool has been shut down.

    Args:
        max_workers (int | None): Maximum number of worker processes. Defaults to
            os.cpu_count() if None.
        initializer (Callable | None): Optional function run in each worker process
            when it starts.
        initargs (tuple): Arguments passed to the initializer.
        name (str): Optional name used only for debugging/logging.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        *,
        initializer: Callable[..., object] | None = None,
        initargs: tuple[object, ...] = (),
        name: str = "ProcessPoolManager",
    ) -> None:
        logger.debug(f"PPM __init__ name={name} id={id(self)}")
        requested_workers = max_workers or (os.cpu_count() or 1)
        self._max_workers: int = max(1, min(requested_workers, MAX_POOL_WORKERS))
        self._initializer = initializer
        self._initargs = initargs
        self._name = name

        self._executor: ProcessPoolExecutor | None = None
        self._lock = threading.RLock()  # protects _executor and _is_shutdown
        self._is_shutdown = False

        self._cancel_shm: shared_memory.SharedMemory | None = None

    @property
    def max_workers(self) -> int:
        """Maximum number of worker processes in the pool."""
        return self._max_workers

    @property
    def name(self) -> str:
        """Logical name of the ProcessPoolManager for debugging/logging."""
        return self._name

    def _ensure_cancel_shm(self) -> shared_memory.SharedMemory:
        """Create the shared-memory cancel flag on first use, if not shut down."""
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError(f"{self._name} has been shut down")

            if self._cancel_shm is None:
                shm = shared_memory.SharedMemory(create=True, size=1)
                # 0 = not cancelled, 1 = cancelled
                shm.buf[0] = 0
                self._cancel_shm = shm

            return self._cancel_shm

    @property
    def cancel_flag_name(self) -> str | None:
        """Name of the shared-memory cancel flag, or None if shut down.

        Callers can pass this name to worker functions so they can open the
        shared memory and cooperatively check for cancellation.
        """
        with self._lock:
            if self._is_shutdown:
                return None

        shm = self._ensure_cancel_shm()
        return shm.name

    def set_cancelled(self) -> None:
        """Set the cancel flag to 1, signalling cooperative cancellation."""
        with self._lock:
            if self._is_shutdown:
                return

            shm = self._cancel_shm or self._ensure_cancel_shm()
            shm.buf[0] = 1

    def clear_cancelled(self) -> None:
        """Reset the cancel flag back to 0."""
        with self._lock:
            if self._cancel_shm is not None:
                self._cancel_shm.buf[0] = 0

    def _ensure_executor(self) -> ProcessPoolExecutor:
        """Create the executor on first use, if not shut down."""
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError(f"{self._name} has been shut down")

            if self._executor is None:
                # noinspection PyTypeChecker
                self._executor = ProcessPoolExecutor(
                    max_workers=self._max_workers,
                    initializer=self._initializer,
                    initargs=self._initargs,
                )

            return self._executor

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit a task to the process pool."""
        executor = self._ensure_executor()
        return executor.submit(fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int = 1,
    ) -> Iterable[Any]:
        """Map over an iterable using the process pool."""
        executor = self._ensure_executor()
        return executor.map(fn, iterable, chunksize=chunksize)

    def warm_up(self, wait: bool = True) -> None:
        """Eagerly start worker processes and optionally run trivial tasks.

        This is useful if you want the cost of spawning processes to happen
        at a controlled time (e.g., on app startup) instead of on the first
        real submit().

        Args:
            wait (bool): If True, submit and wait for trivial tasks to complete
                in each worker process. This ensures that all workers are fully
                initialized and ready to accept real tasks. If False, only starts
                the processes without waiting for task completion.
        """
        start_time = time.time()
        logger.debug(f"PPM warm_up name={self._name} id={id(self)}")
        executor = self._ensure_executor()
        self._ensure_cancel_shm()

        if not wait:
            return

        # Submit trivial no-op tasks to ensure all workers are started, we use
        # 2 x max_workers to try to ensure all workers get at least one task since they are fast
        futures = [executor.submit(_noop) for _ in range(self._max_workers * 2)]
        for f in futures:
            with contextlib.suppress(Exception):
                f.result()

        elapsed = time.time() - start_time
        logger.debug(
            f"PPM warm_up name={self._name} id={id(self)} COMPLETED in {elapsed:.2f} seconds"
        )

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Explicitly shut down the process pool.

        After shutdown, the manager cannot be reused.
        """
        with self._lock:
            self._is_shutdown = True
            executor = self._executor
            if executor is not None:
                with contextlib.suppress(Exception):
                    executor.shutdown(wait=wait, cancel_futures=cancel_futures)
                self._executor = None
            if self._cancel_shm is not None:
                with contextlib.suppress(Exception):
                    self._cancel_shm.close()
                    self._cancel_shm.unlink()
                self._cancel_shm = None

    def __enter__(self) -> "ProcessPoolManager":
        """Enter context manager, returning self.

        Allows using the manager in a 'with' statement for automatic cleanup.
        """
        self._ensure_executor()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit context manager, shutting down the process pool."""
        self.shutdown(wait=True, cancel_futures=False)

    def __del__(self) -> None:
        """Best-effort cleanup if user code forgets to call shutdown().

        Note: __del__ is not guaranteed to run at interpreter shutdown, so
        you should still call shutdown() or use the manager as a context manager.
        """
        with contextlib.suppress(Exception):
            self.shutdown(wait=False, cancel_futures=True)
