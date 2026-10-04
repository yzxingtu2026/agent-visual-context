"""感知层边界：检测、跟踪与关系推理。"""

from __future__ import annotations

from .base import Detector, RelationReasoner, Tracker
from .mock import (
    MockRelationReasoner,
    MockTarget,
    MockTracker,
    RelationRule,
    ScriptedDetector,
    StaticSceneDetector,
)
from .relate_anything import (
    DEFAULT_VOCABULARY,
    RawTriplet,
    RelateAnythingBackend,
    RelateAnythingReasoner,
    RelateAnythingSettings,
    RelationMetrics,
    RelationStats,
    RelSggBackend,
)
from .yolo_world import (
    DetectionMetrics,
    DetectionStats,
    RawDetection,
    UltralyticsYoloWorldBackend,
    YoloWorldBackend,
    YoloWorldDetector,
    YoloWorldSettings,
)

__all__ = [
    "DEFAULT_VOCABULARY",
    "DetectionMetrics",
    "DetectionStats",
    "Detector",
    "MockRelationReasoner",
    "MockTarget",
    "MockTracker",
    "RawDetection",
    "RawTriplet",
    "RelSggBackend",
    "RelateAnythingBackend",
    "RelateAnythingReasoner",
    "RelateAnythingSettings",
    "RelationMetrics",
    "RelationReasoner",
    "RelationRule",
    "RelationStats",
    "ScriptedDetector",
    "StaticSceneDetector",
    "Tracker",
    "UltralyticsYoloWorldBackend",
    "YoloWorldBackend",
    "YoloWorldDetector",
    "YoloWorldSettings",
]
