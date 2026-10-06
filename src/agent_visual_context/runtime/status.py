"""运行时状态与降级信息。

视觉处理必须可降级：单个组件失败时记录状态并继续，不得阻塞流水线或上层链路。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ..domain import utc_now


class ComponentState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


class PipelineState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


class ComponentHealth(BaseModel):
    state: ComponentState = ComponentState.HEALTHY
    model_version: str = "unknown"
    consecutive_failures: int = 0
    last_error: str | None = None
    last_ok_at: datetime | None = None
    last_latency_ms: float | None = None
    calls: int = 0
    failures: int = 0


class PipelineStatus(BaseModel):
    """流水线运行状态快照，可被 CLI、日志和上层 Agent 读取。"""

    state: PipelineState = PipelineState.IDLE
    scene_id: str = ""
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    frames_processed: int = 0
    observations_emitted: int = 0
    events_emitted: int = 0
    dropped_items: int = 0
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_degraded(self) -> bool:
        return any(
            component.state in (ComponentState.DEGRADED, ComponentState.FAILED)
            for component in self.components.values()
        )

    def register(self, name: str, *, model_version: str = "unknown") -> ComponentHealth:
        health = self.components.get(name)
        if health is None:
            health = ComponentHealth(model_version=model_version)
            self.components[name] = health
        elif model_version != "unknown":
            health.model_version = model_version
        return health

    def mark_ok(self, name: str, *, now: datetime) -> None:
        health = self.register(name)
        health.state = ComponentState.HEALTHY
        health.consecutive_failures = 0
        health.last_error = None
        health.last_ok_at = now
        health.calls += 1

    def mark_failure(self, name: str, *, error: str, now: datetime, fatal: bool = False) -> None:
        health = self.register(name)
        health.consecutive_failures += 1
        health.calls += 1
        health.failures += 1
        health.last_error = error
        health.state = ComponentState.FAILED if fatal else ComponentState.DEGRADED
        self.last_error = f"{name}: {error}"
        self.updated_at = now

    def disable(self, name: str) -> None:
        health = self.register(name)
        health.state = ComponentState.DISABLED

    def refresh(self, *, now: datetime) -> None:
        self.updated_at = now
        if self.state == PipelineState.FAILED:
            return
        self.state = PipelineState.DEGRADED if self.is_degraded else self.state
