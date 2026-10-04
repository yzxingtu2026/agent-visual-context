"""输入源边界：帧采集与回放。"""

from __future__ import annotations

from .base import AbstractFrameSource, FrameSource
from .camera import CameraBackend, CameraSource, OpenCVCameraBackend
from .image import ImageFrameSource
from .loader import IMAGE_SUFFIXES, VIDEO_SUFFIXES, frame_source_from_path
from .mock import ScriptedFrameSource, synthetic_frames
from .sampling import plan_sampling
from .video import VideoFrameSource

__all__ = [
    "IMAGE_SUFFIXES",
    "VIDEO_SUFFIXES",
    "AbstractFrameSource",
    "CameraBackend",
    "CameraSource",
    "FrameSource",
    "ImageFrameSource",
    "OpenCVCameraBackend",
    "ScriptedFrameSource",
    "VideoFrameSource",
    "frame_source_from_path",
    "plan_sampling",
    "synthetic_frames",
]
