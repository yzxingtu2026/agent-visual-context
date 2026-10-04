"""运行时层边界：流水线编排、事件总线、生命周期与降级状态。"""

from __future__ import annotations

from .bus import EventBus, EventListener
from .factory import (
    DEFAULT_RELATION_RULES,
    DEFAULT_TARGETS,
    build_mock_pipeline,
    build_offline_pipeline,
)
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
    "Pipeline",
    "PipelineState",
    "PipelineStatus",
    "RunResult",
    "build_mock_pipeline",
    "build_offline_pipeline",
]
