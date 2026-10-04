"""输入源边界：帧采集与回放。"""

from __future__ import annotations

from .base import AbstractFrameSource, FrameSource
from .mock import ScriptedFrameSource, synthetic_frames

__all__ = ["AbstractFrameSource", "FrameSource", "ScriptedFrameSource", "synthetic_frames"]
