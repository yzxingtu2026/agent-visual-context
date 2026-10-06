"""运行时层边界：流水线编排、事件总线、生命周期与降级状态。"""

from __future__ import annotations

from .buffer import LatestFrameBuffer
from .bus import EventBus, EventListener
from .factory import (
    DEFAULT_RELATION_RULES,
    DEFAULT_TARGETS,
    build_camera_source,
    build_detector,
    build_face_analyzer,
    build_live_runtime,
    build_mock_pipeline,
    build_offline_pipeline,
    build_tracker,
)
from .live import LiveRunResult, LiveRuntime
from .metrics import LiveMetrics, MetricsSnapshot
from .pipeline import FrameResult, Pipeline, RunResult
from .status import ComponentHealth, ComponentState, PipelineState, PipelineStatus

__all__ = [
    "DEFAULT_RELATION_RULES",
    "DEFAULT_TARGETS",
    "ComponentHealth",
    "ComponentState",
    "EventBus",
    "EventListener",
    "FrameResult",
    "LatestFrameBuffer",
    "LiveMetrics",
    "LiveRunResult",
    "LiveRuntime",
    "MetricsSnapshot",
    "Pipeline",
    "PipelineState",
    "PipelineStatus",
    "RunResult",
    "build_camera_source",
    "build_detector",
    "build_face_analyzer",
    "build_live_runtime",
    "build_mock_pipeline",
    "build_offline_pipeline",
    "build_tracker",
]
