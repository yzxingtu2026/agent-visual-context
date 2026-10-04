"""实时视觉 Sidecar 循环（PoC-5）。

`LiveRuntime` 把"离线跑到数据源耗尽"的 `Pipeline.run()` 升级为**持续运行的实时链路**，
用于本地摄像头场景。核心设计与降级保证：

- **采集/消费解耦**：后台采集线程持续 `source.read()` 写入 `LatestFrameBuffer`（有界、
  丢旧保最新），主消费循环按缓冲产出节奏取最新帧交给 `Pipeline.process_frame()`。
  高帧率采集与低频（1~2 FPS）推理互不阻塞，缓冲永不无限堆积。
- **复用既有降级**：单帧处理仍走 `Pipeline.process_frame()`，检测/跟踪/关系推理各自带
  超时与异常捕获，任一组件失败只降级该组件、跳过该帧，不中断循环。推理超时由适配器的
  `*_timeout_seconds` 覆盖；消费循环再对兜底异常做一次捕获，确保循环不会因单帧崩溃退出。
- **视觉异常不阻塞宿主**：采集线程为 daemon 线程，摄像头打不开/读帧连续失败时只把
  `source` 组件标记为降级并停止采集，`run()` 正常返回 `degraded` 状态，绝不向上抛出，
  从而不阻塞大屏渲染、麦克风采集、语音 WebSocket 与 TTS 播放等宿主链路。
- **生命周期**：`start()`/`stop()` 管理采集线程与设备；`run()` 是一次有界的运行
  （按 `duration_seconds` 或 `max_frames` 退出）；也支持上下文管理器。
- **可观测**：`LiveMetrics` 累积有效 FPS、端到端延迟、丢帧率与模型失败率，
  周期性写结构化日志；健康状态统一汇入 `Pipeline.status`。

摄像头循环只写在本模块，**不得**下沉进模型适配器或 CLI。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..domain import Frame, Observation, Snapshot, utc_now
from ..logging_setup import get_logger
from .buffer import LatestFrameBuffer
from .metrics import LiveMetrics, MetricsSnapshot
from .pipeline import Pipeline
from .status import PipelineState, PipelineStatus

logger = get_logger("runtime.live")


@dataclass(slots=True)
class LiveRunResult:
    """一次实时运行的汇总结果。"""

    frames_processed: int = 0
    observations: list[Observation] = field(default_factory=list)
    metrics: MetricsSnapshot | None = None
    status: PipelineStatus = field(default_factory=PipelineStatus)
    snapshot: Snapshot | None = None

    @property
    def degraded(self) -> bool:
        return self.status.is_degraded or self.status.state == PipelineState.DEGRADED


class LiveRuntime:
    """驱动摄像头实时链路的 Sidecar 运行时。"""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        buffer: LatestFrameBuffer | None = None,
        metrics: LiveMetrics | None = None,
        clock: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        poll_interval: float = 0.05,
        max_capture_failures: int = 5,
        join_timeout: float = 2.0,
        log_every_frames: int = 20,
    ) -> None:
        self._pipeline = pipeline
        self._source = pipeline.source
        self._clock = clock
        self._monotonic = monotonic
        self._poll_interval = max(0.001, poll_interval)
        self._max_capture_failures = max(1, max_capture_failures)
        self._join_timeout = join_timeout
        self._log_every_frames = max(1, log_every_frames)
        self.buffer = buffer if buffer is not None else LatestFrameBuffer(maxsize=1)
        self.metrics = metrics if metrics is not None else LiveMetrics(clock=clock)

        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._capture_failed = False
        self._running = False

    @property
    def pipeline(self) -> Pipeline:
        """被驱动的流水线；`api` 层通过它复用同一时间线与摘要器。"""
        return self._pipeline

    @property
    def status(self) -> PipelineStatus:
        return self._pipeline.status

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def capture_failed(self) -> bool:
        """采集线程是否因连续失败而终止（视觉降级的强信号）。"""
        return self._capture_failed

    def start(self) -> None:
        """打开摄像头并启动采集线程。设备打开失败会抛 `FrameSourceError` 由上层降级。"""
        if self._running:
            return
        self._stop_event.clear()
        self._capture_failed = False
        self._source.open()
        self.metrics.start(now=self._clock())
        self._pipeline.status.state = PipelineState.RUNNING
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="avc-camera-capture",
            daemon=True,
        )
        self._capture_thread.start()
        self._running = True
        logger.info("实时视觉链路启动 source=%s", self._source.source_id)

    def stop(self) -> None:
        """停止采集线程并释放设备；幂等。"""
        if not self._running:
            return
        self._stop_event.set()
        thread = self._capture_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._join_timeout)
        self._capture_thread = None
        self._running = False
        try:
            self._source.close()
        except Exception as exc:  # 释放失败不应影响停止流程
            logger.warning("摄像头释放异常（忽略）：%s", exc)
        self._finalize_state()
        logger.info("实时视觉链路停止")

    def run(
        self,
        *,
        duration_seconds: float | None = None,
        max_frames: int | None = None,
    ) -> LiveRunResult:
        """运行实时链路直到达到时长/帧数上限或采集降级退出，然后停止。

        始终返回 `LiveRunResult`（不抛出视觉异常），供 CLI 与宿主链路读取健康状态。
        """
        result = LiveRunResult(status=self._pipeline.status)
        try:
            self.start()
        except Exception as exc:
            # 摄像头打不开/被占用：降级返回，绝不向上抛出，避免阻塞宿主链路。
            self._capture_failed = True
            self._pipeline.status.mark_failure(
                "source", error=str(exc), now=self._clock(), fatal=False
            )
            self._finalize_state()
            result.metrics = self.metrics.snapshot(now=self._clock())
            result.status = self._pipeline.status
            logger.error("摄像头启动失败，实时视觉链路降级：%s", exc)
            return result

        deadline = None if duration_seconds is None else self._monotonic() + duration_seconds
        try:
            self._consume(result, deadline=deadline, max_frames=max_frames)
        finally:
            self.stop()
        result.metrics = self.metrics.snapshot(now=self._clock())
        result.snapshot = self._pipeline.summarizer.build(
            self._pipeline.timeline,
            now=self._clock(),
            degraded=self._pipeline.status.is_degraded,
        )
        result.status = self._pipeline.status
        self._log_metrics(result.metrics)
        return result

    def __enter__(self) -> LiveRuntime:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    # -- 内部实现 ---------------------------------------------------------

    def _capture_loop(self) -> None:
        """后台采集线程：持续读帧写入有界缓冲；连续失败则降级退出。"""
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                frame = self._source.read()
            except Exception as exc:
                consecutive_failures += 1
                self._pipeline.status.mark_failure("source", error=str(exc), now=self._clock())
                logger.warning(
                    "摄像头读帧失败（%s/%s）：%s",
                    consecutive_failures,
                    self._max_capture_failures,
                    exc,
                )
                if consecutive_failures >= self._max_capture_failures:
                    self._capture_failed = True
                    logger.error("摄像头连续读帧失败达上限，停止采集并降级")
                    break
                continue

            if frame is None:
                # 实时流一般不耗尽；返回 None 视为设备流结束。
                logger.warning("摄像头数据流结束，停止采集")
                break

            consecutive_failures = 0
            self.buffer.put(frame)
            self.metrics.record_captured()

    def _consume(
        self,
        result: LiveRunResult,
        *,
        deadline: float | None,
        max_frames: int | None,
    ) -> None:
        """主消费循环：按缓冲产出取最新帧处理，直到退出条件满足。"""
        last_dropped = 0
        while not self._stop_event.is_set():
            if deadline is not None and self._monotonic() >= deadline:
                break
            if max_frames is not None and result.frames_processed >= max_frames:
                break

            frame = self.buffer.get(timeout=self._poll_interval)
            self._sync_dropped(last_dropped)
            last_dropped = self.buffer.dropped_total

            if frame is None:
                if self._capture_stalled():
                    break
                continue

            self._process(frame, result)

    def _process(self, frame: Frame, result: LiveRunResult) -> None:
        """处理单帧：捕获兜底异常，确保循环不因单帧崩溃退出。"""
        try:
            frame_result = self._pipeline.process_frame(frame)
        except Exception as exc:
            # process_frame 内部已对组件降级；这里是最后一道兜底，防止实时循环中断。
            self._pipeline.status.mark_failure("pipeline", error=str(exc), now=self._clock())
            logger.exception("实时帧处理异常，已降级跳过该帧")
            self.metrics.record_processed(latency_seconds=0.0, degraded=True)
            return

        latency = (self._clock() - frame.captured_at).total_seconds()
        degraded = bool(frame_result.degraded_components)
        self.metrics.record_processed(latency_seconds=max(0.0, latency), degraded=degraded)
        result.frames_processed += 1
        result.observations.extend(frame_result.observations)

        if result.frames_processed % self._log_every_frames == 0:
            self._log_metrics(self.metrics.snapshot(now=self._clock()))

    def _sync_dropped(self, last_dropped: int) -> None:
        current = self.buffer.dropped_total
        delta = current - last_dropped
        if delta > 0:
            self.metrics.record_dropped(delta)

    def _capture_stalled(self) -> bool:
        """采集线程已死且缓冲无帧时，消费循环无更多可处理数据。"""
        thread = self._capture_thread
        return thread is not None and not thread.is_alive() and len(self.buffer) == 0

    def _finalize_state(self) -> None:
        status = self._pipeline.status
        now = self._clock()
        if self._capture_failed:
            status.mark_failure("source", error="摄像头采集连续失败，已停止", now=now, fatal=False)
        status.frames_processed = max(status.frames_processed, 0)
        status.dropped_items = self.metrics.snapshot(now=now).frames_dropped
        status.refresh(now=now)
        if status.state != PipelineState.FAILED:
            status.state = PipelineState.DEGRADED if status.is_degraded else PipelineState.STOPPED

    def _log_metrics(self, snapshot: MetricsSnapshot) -> None:
        logger.info(
            "实时指标 运行=%.1fs 采集=%s 处理=%s 丢弃=%s 有效FPS=%.2f 丢帧率=%.2f "
            "失败率=%.2f 平均延迟=%.3fs",
            snapshot.uptime_seconds,
            snapshot.frames_captured,
            snapshot.frames_processed,
            snapshot.frames_dropped,
            snapshot.effective_fps,
            snapshot.drop_rate,
            snapshot.failure_rate,
            snapshot.mean_latency_seconds,
        )
