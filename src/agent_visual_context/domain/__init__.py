"""领域层：视觉上下文的纯数据模型，不依赖任何具体实现。"""

from __future__ import annotations

from .models import (
    BBox,
    Detection,
    EpistemicStatus,
    Event,
    Frame,
    Observation,
    RawFrameData,
    Relation,
    Snapshot,
    TrackedObject,
    TrackRef,
    new_id,
    utc_now,
)

__all__ = [
    "BBox",
    "Detection",
    "EpistemicStatus",
    "Event",
    "Frame",
    "Observation",
    "RawFrameData",
    "Relation",
    "Snapshot",
    "TrackRef",
    "TrackedObject",
    "new_id",
    "utc_now",
]
