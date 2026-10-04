"""可替换的输入源实现。骨架阶段只提供合成/脚本数据源，不依赖 OpenCV。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta

from ..domain import Frame
from .base import AbstractFrameSource


def synthetic_frames(
    count: int,
    *,
    source_id: str = "mock-source",
    width: int = 640,
    height: int = 480,
    start: datetime,
    interval: timedelta = timedelta(milliseconds=500),
) -> list[Frame]:
    """生成不含真实像素的占位帧，用于离线 PoC 与测试。"""
    if count < 0:
        msg = "count 不能为负数"
        raise ValueError(msg)
    return [
        Frame(
            frame_id=f"{source_id}-{index:05d}",
            source_id=source_id,
            captured_at=start + interval * index,
            width=width,
            height=height,
            data=None,
            metadata={"synthetic": True, "index": index},
        )
        for index in range(count)
    ]


class ScriptedFrameSource(AbstractFrameSource):
    """按脚本回放固定帧序列的数据源，可注入任意测试替身帧。"""

    def __init__(self, frames: Sequence[Frame], *, source_id: str | None = None) -> None:
        resolved_id = source_id if source_id is not None else _resolve_source_id(frames)
        super().__init__(resolved_id)
        self._frames = list(frames)
        self._cursor = 0
        self.open_count = 0
        self.close_count = 0

    def frames(self) -> Iterator[Frame]:
        """迭代剩余帧（自动打开数据源）。"""
        self.open()
        while (frame := self.read()) is not None:
            yield frame

    def _do_open(self) -> None:
        self.open_count += 1

    def _do_read(self) -> Frame | None:
        if self._cursor >= len(self._frames):
            return None
        frame = self._frames[self._cursor]
        self._cursor += 1
        return frame

    def _do_close(self) -> None:
        self.close_count += 1


def _resolve_source_id(frames: Sequence[Frame]) -> str:
    return frames[0].source_id if frames else "mock-source"
