"""核心领域模型。

约束：本模块只依赖标准库与 Pydantic，不得引入 OpenCV、具体模型或 Agent 框架。
图像等不可序列化的原始数据以 `RawFrameData`（对领域层透明）承载，由输入/感知适配层解释。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# 对领域层透明的原始帧数据（例如 numpy 数组、图像路径或测试替身）。
RawFrameData = Any


def utc_now() -> datetime:
    """统一的时间源，便于测试中替换。"""
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class EpistemicStatus(StrEnum):
    """认知状态：视觉观察不能直接成为业务事实。"""

    VISUAL_OBSERVATION = "visual-observation"
    RULE_EVENT = "rule-event"
    USER_ASSERTION = "user-assertion"


class BBox(BaseModel):
    """像素坐标系下的目标框。"""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


class Frame(BaseModel):
    """一次采样的输入帧。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: str
    source_id: str
    captured_at: datetime
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    data: RawFrameData = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Detection(BaseModel):
    """检测器输出的单个目标。"""

    model_config = ConfigDict(frozen=True)

    label: str
    bbox: BBox
    confidence: float = Field(ge=0, le=1)
    model_version: str = "unknown"
    detected_at: datetime


class TrackedObject(BaseModel):
    """跨帧稳定的跟踪目标。"""

    model_config = ConfigDict(frozen=True)

    track_id: str
    label: str
    bbox: BBox
    confidence: float = Field(ge=0, le=1)
    first_seen: datetime
    last_seen: datetime
    model_version: str = "unknown"

    @property
    def duration_seconds(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds()


class TrackRef(BaseModel):
    """观察中引用的一端（主语或宾语）。"""

    model_config = ConfigDict(frozen=True)

    track_id: str
    label: str


class Relation(BaseModel):
    """关系推理器输出的单帧关系。"""

    model_config = ConfigDict(frozen=True)

    subject: TrackRef
    predicate: str
    target: TrackRef
    confidence: float = Field(ge=0, le=1)
    model_version: str = "unknown"
    observed_at: datetime

    @property
    def key(self) -> tuple[str, str, str]:
        """用于去抖与持续性统计的稳定键。"""
        return (self.subject.track_id, self.predicate, self.target.track_id)


class Observation(BaseModel):
    """带时间戳、置信度、来源、模型版本与有效期的视觉观察。"""

    model_config = ConfigDict(frozen=True)

    observation_id: str = Field(default_factory=lambda: new_id("obs"))
    scene_id: str
    subject: TrackRef
    predicate: str
    target: TrackRef | None = None
    confidence: float = Field(ge=0, le=1)
    observed_at: datetime
    expires_at: datetime
    source_id: str
    model_version: str = "unknown"
    epistemic_status: EpistemicStatus = EpistemicStatus.VISUAL_OBSERVATION

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def ttl_seconds(self, now: datetime) -> float:
        return (self.expires_at - now).total_seconds()


class Event(BaseModel):
    """规则派生事件（例如迎宾候选），与视觉观察分型。"""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    scene_id: str
    kind: str
    observed_at: datetime
    expires_at: datetime
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_id: str = "policy"
    epistemic_status: EpistemicStatus = EpistemicStatus.RULE_EVENT
    payload: dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class Snapshot(BaseModel):
    """有界时间窗口内的场景摘要，用于注入 Agent 话轮上下文。"""

    scene_id: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    observations: list[Observation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    degraded: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.window_end - self.window_start).total_seconds()
