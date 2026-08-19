"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time

from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class _ConcurrencyController:
    """AIMD controller for the in-flight group cap (``max_concurrent``).

    A static cap is a poor fit when the bottleneck is the rollout env (SIEM
    query latency swings with cache warmth and trajectory length): too high
    stampedes the backend, too low starves the engines. This tracks group
    completion latency and treats a rise over the recently-achievable baseline
    as congestion — additive-increase the cap while latency is healthy,
    multiplicative-decrease it when latency inflates. The baseline slowly
    relaxes upward so a genuine regime change (heavier trajectories) becomes
    the new normal instead of throttling forever. Bounded to
    [one batch of groups, batch * multiplier_max].
    """

    def __init__(self, initial, mc_min, mc_max, *, congestion_ratio, decrease_factor,
                 increase_step, interval_s, min_samples, baseline_relax, alpha=0.2):
        self.mc = int(initial)
        self.mc_min, self.mc_max = int(mc_min), int(mc_max)
        self.congestion_ratio = congestion_ratio
        self.decrease_factor = decrease_factor
        self.increase_step = int(increase_step)
        self.interval_s = interval_s
        self.min_samples = int(min_samples)
        self.baseline_relax = baseline_relax
        self.alpha = alpha
        self.lat_ewma = None
        self.lat_baseline = None
        self._n_since = 0
        self._last_adjust = 0.0

    def record(self, latency: float) -> None:
        self.lat_ewma = latency if self.lat_ewma is None else (
            (1 - self.alpha) * self.lat_ewma + self.alpha * latency
        )
        self._n_since += 1

    def maybe_adjust(self, now: float):
        """Run the AIMD step if the interval elapsed and enough groups landed.

        Returns a (prev, new, congested, lat_ewma, baseline) tuple when it acted,
        else None.
        """
        if now - self._last_adjust < self.interval_s:
            return None
        if self.lat_ewma is None or self._n_since < self.min_samples:
            self._last_adjust = now
            return None
        self._last_adjust = now
        # Baseline = recently-achievable latency: snap down to any faster reading,
        # else relax up slowly so a sustained-higher regime is eventually accepted.
        if self.lat_baseline is None or self.lat_ewma < self.lat_baseline:
            self.lat_baseline = self.lat_ewma
        else:
            self.lat_baseline *= (1 + self.baseline_relax)
        congested = self.lat_ewma > self.lat_baseline * self.congestion_ratio
        prev = self.mc
        if congested:
            self.mc = max(self.mc_min, int(self.mc * self.decrease_factor))
        else:
            self.mc = min(self.mc_max, self.mc + self.increase_step)
        self._n_since = 0
        return (prev, self.mc, congested, self.lat_ewma, self.lat_baseline)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        # AIMD dynamic-concurrency controller (opt-in); None => static cap.
        self._dc: _ConcurrencyController | None = None

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        # Opt-in AIMD dynamic concurrency, in GROUP units (this loop schedules
        # one task per group). Off by default => static cap (unchanged behavior).
        if getattr(self.args, "fully_async_dynamic_concurrency", False):
            bs = max(1, int(getattr(self.args, "rollout_batch_size", 1)))
            mc_min = bs  # never below one batch of groups, so a step can still form
            mult_max = float(
                getattr(self.args, "fully_async_concurrent_multiplier_max", None)
                or getattr(self.args, "fully_async_concurrent_multiplier", None)
                or 1.0
            )
            mc_max = max(mc_min, int(bs * mult_max))
            self._dc = _ConcurrencyController(
                initial=min(self.concurrency, mc_max),
                mc_min=mc_min,
                mc_max=mc_max,
                congestion_ratio=float(getattr(self.args, "fully_async_concurrency_congestion_ratio", None) or 1.5),
                decrease_factor=float(getattr(self.args, "fully_async_concurrency_decrease_factor", None) or 0.7),
                increase_step=int(getattr(self.args, "fully_async_concurrency_increase_step", None) or 1),
                interval_s=float(getattr(self.args, "fully_async_concurrency_control_interval_s", None) or 30.0),
                min_samples=int(getattr(self.args, "fully_async_concurrency_min_samples", None) or 2),
                baseline_relax=float(getattr(self.args, "fully_async_concurrency_baseline_relax", None) or 0.02),
            )
            max_concurrent = self._dc.mc
            logger.info(
                "fully-async DYNAMIC concurrency ON (group units): init=%d bounds=[%d,%d] "
                "congestion_ratio=%.2f decrease=%.2f increase_step=%d interval=%.0fs",
                self._dc.mc, mc_min, mc_max, self._dc.congestion_ratio,
                self._dc.decrease_factor, self._dc.increase_step, self._dc.interval_s,
            )

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                # Dynamic cap: pick up the controller's latest value so the
                # top-up below uses the live cap. A decrease never cancels
                # in-flight groups — refill just pauses until they drain.
                if self._dc is not None:
                    max_concurrent = self._dc.mc

                # Top up.
                while len(active_tasks) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid, time.time()))
                        active_tasks.add(task)

                # AIMD: adjust the concurrency cap on its own interval and log it.
                if self._dc is not None:
                    adj = self._dc.maybe_adjust(time.time())
                    if adj is not None:
                        prev, new_mc, congested, le, lb = adj
                        max_concurrent = new_mc
                        logger.info(
                            "fully-async concurrency: mc=%d prev=%d congested=%s "
                            "lat_ewma=%.1fs baseline=%.1fs inflight=%d",
                            new_mc, prev, congested, le, lb, len(active_tasks),
                        )

                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int, launch_t: float):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            # Aborted group → requeue, don't ship to training.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            # Feed group latency to the AIMD controller. Completed groups only:
            # aborted groups short-circuit fast (weight-update pause) and would
            # drag the baseline down, making healthy latency look congested.
            if self._dc is not None:
                self._dc.record(time.time() - launch_t)
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull whatever's done.
        drained = 0
        for gid, group in worker.get_completed_groups():
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)[:target]
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return out


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
