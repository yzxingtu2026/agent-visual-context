"""实时链路运行指标（PoC-5）。

`LiveMetrics` 累积摄像头实时链路的关键可观测指标，供健康状态、结构化日志与
上层 Agent 查询读取。所有计数线程安全（采集线程与消费线程并发更新）。

指标口径：
- `frames_captured`：采集线程成功抓取并写入缓冲的帧数；
- `frames_dropped`：缓冲因"丢旧保最新"而丢弃的帧数（来自 `LatestFrameBuffer`）；
- `frames_processed`：消费线程实际完成推理处理的帧数；
- `effective_fps`：`frames_processed / 运行时长`，即真实有效处理帧率；
- `drop_rate`：`frames_dropped / frames_captured`，反映采集与处理速率的失配程度；
- `model_failures` / `failure_rate`：处理过程中出现组件降级的帧数与占比；
- `latency`：端到端延迟（帧采集时刻到处理完成），记录 min/max/mean。

CPU 与内存占用（`cpu_percent` / `rss_mb`）为**预留字段**：目标设备（Windows 一体机）
真机压测在后续 Issue 执行时由基准脚本填充，本模块不引入 `psutil` 等额外依赖。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ..domain import utc_now


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """某一时刻的只读指标快照，可安全地跨线程/序列化传递。"""

    uptime_seconds: float
    frames_captured: int
    frames_processed: int
    frames_dropped: int
    model_failures: int
    effective_fps: float
    drop_rate: float
    failure_rate: float
    mean_latency_seconds: float
    min_latency_seconds: float | None
    max_latency_seconds: float | None
    cpu_percent: float | None
    rss_mb: float | None


class LiveMetrics:
    """线程安全的实时链路指标累积器。"""

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._started_at: datetime | None = None
        self._frames_captured = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._model_failures = 0
        self._total_latency = 0.0
        self._min_latency: float | None = None
        self._max_latency: float | None = None
        self._cpu_percent: float | None = None
        self._rss_mb: float | None = None

    def start(self, *, now: datetime | None = None) -> None:
        """重置计数并记录起始时刻。"""
        with self._lock:
            self._started_at = now if now is not None else self._now()
            self._frames_captured = 0
            self._frames_processed = 0
            self._frames_dropped = 0
            self._model_failures = 0
            self._total_latency = 0.0
            self._min_latency = None
            self._max_latency = None

    def record_captured(self, count: int = 1) -> None:
        with self._lock:
            self._frames_captured += count

    def record_dropped(self, count: int) -> None:
        with self._lock:
            self._frames_dropped += count

    def record_processed(self, *, latency_seconds: float, degraded: bool = False) -> None:
        with self._lock:
            self._frames_processed += 1
            self._total_latency += latency_seconds
            self._min_latency = (
                latency_seconds
                if self._min_latency is None
                else min(self._min_latency, latency_seconds)
            )
            self._max_latency = (
                latency_seconds
                if self._max_latency is None
                else max(self._max_latency, latency_seconds)
            )
            if degraded:
                self._model_failures += 1

    def set_resource_usage(self, *, cpu_percent: float | None, rss_mb: float | None) -> None:
        """由外部基准脚本填充 CPU/内存占用；默认保持 `None`。"""
        with self._lock:
            self._cpu_percent = cpu_percent
            self._rss_mb = rss_mb

    def uptime_seconds(self, *, now: datetime | None = None) -> float:
        with self._lock:
            if self._started_at is None:
                return 0.0
            moment = now if now is not None else self._now()
            return max(0.0, (moment - self._started_at).total_seconds())

    def snapshot(self, *, now: datetime | None = None) -> MetricsSnapshot:
        """生成当前只读指标快照。"""
        with self._lock:
            uptime = self.uptime_seconds(now=now)
            effective_fps = self._frames_processed / uptime if uptime > 0 else 0.0
            drop_rate = (
                self._frames_dropped / self._frames_captured if self._frames_captured else 0.0
            )
            failure_rate = (
                self._model_failures / self._frames_processed if self._frames_processed else 0.0
            )
            mean_latency = (
                self._total_latency / self._frames_processed if self._frames_processed else 0.0
            )
            return MetricsSnapshot(
                uptime_seconds=uptime,
                frames_captured=self._frames_captured,
                frames_processed=self._frames_processed,
                frames_dropped=self._frames_dropped,
                model_failures=self._model_failures,
                effective_fps=effective_fps,
                drop_rate=drop_rate,
                failure_rate=failure_rate,
                mean_latency_seconds=mean_latency,
                min_latency_seconds=self._min_latency,
                max_latency_seconds=self._max_latency,
                cpu_percent=self._cpu_percent,
                rss_mb=self._rss_mb,
            )

    def _now(self) -> datetime:
        return self._clock()
