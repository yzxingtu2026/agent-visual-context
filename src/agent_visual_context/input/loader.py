"""输入源装配：按路径类型分发到图片/视频适配器。

CLI、示例与测试统一通过 `frame_source_from_path()` 把离线素材路径
（单张图片、图片目录或视频文件）转换为 `FrameSource`，避免各处重复判断扩展名。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..domain import utc_now
from ..errors import FrameSourceError
from .base import FrameSource
from .image import ImageFrameSource
from .video import VideoFrameSource

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)
VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


def frame_source_from_path(
    path: Path | str,
    *,
    source_id: str | None = None,
    target_fps: float = 2.0,
    start: datetime | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> FrameSource:
    """按路径类型构造输入源。

    - 目录：按文件名顺序回放其中的图片（空目录视为空素材错误）；
    - 图片文件：单帧输入源；
    - 视频文件：按 `target_fps` 采样的视频输入源；
    - 其他类型：抛出 `FrameSourceError`。
    """
    resolved = Path(path)
    if not resolved.exists():
        msg = f"输入素材不存在：{resolved}"
        raise FrameSourceError(msg)

    if resolved.is_dir():
        paths = sorted(item for item in resolved.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            msg = f"目录中没有可用图片（支持 {sorted(IMAGE_SUFFIXES)}）：{resolved}"
            raise FrameSourceError(msg)
        return ImageFrameSource(
            paths,
            source_id=source_id or resolved.name,
            target_fps=target_fps,
            start=start,
            clock=clock,
        )

    suffix = resolved.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return ImageFrameSource(
            [resolved], source_id=source_id, target_fps=target_fps, start=start, clock=clock
        )
    if suffix in VIDEO_SUFFIXES:
        return VideoFrameSource(
            resolved, source_id=source_id, target_fps=target_fps, start=start, clock=clock
        )

    msg = f"不支持的素材类型（{suffix or '无扩展名'}）：{resolved}"
    raise FrameSourceError(msg)
