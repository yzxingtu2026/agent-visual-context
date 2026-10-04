"""视频文件输入源：按采样策略回放 -> 统一 `Frame`。

时间戳与编号约定（与图片源共用同一套语义）：
- `frame_id = "{source_id}-{原始帧编号:05d}"`，metadata 中保留原始帧号与采样序号；
- `captured_at = start + 原始帧编号 / 源帧率`，即视频内相对时间映射到统一时间轴；
- 未显式给定 `start` 时，以“视频末帧即打开时刻”为锚点回放；
- 采样由 `input.sampling.plan_sampling` 决定：源帧率高于 `target_fps` 时等间隔抽帧，否则逐帧全选。

解码依赖 OpenCV，仅在 `open()` 时惰性加载；numpy 像素数据放入 `Frame.data`，
OpenCV 类型不进入领域层。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..domain import Frame, utc_now
from ..errors import FrameSourceError
from ..logging_setup import get_logger
from ._opencv import require_cv2
from .base import AbstractFrameSource
from .sampling import plan_sampling

logger = get_logger("input.video")


class VideoFrameSource(AbstractFrameSource):
    """从视频文件按目标帧率采样帧的输入源。"""

    def __init__(
        self,
        path: Path | str,
        *,
        source_id: str | None = None,
        target_fps: float = 2.0,
        start: datetime | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        resolved = Path(path)
        if target_fps <= 0:
            msg = f"target_fps 必须为正数，收到 {target_fps}"
            raise FrameSourceError(msg)
        super().__init__(source_id if source_id is not None else resolved.stem)
        self._path = resolved
        self._target_fps = target_fps
        self._start = start
        self._clock = clock
        # cv2.VideoCapture 无公开类型别名，运行时由 OpenCV 提供
        self._capture: Any = None
        self._plan: list[int] = []
        self._plan_position = 0
        self._next_raw_index = 0
        self._source_fps = 0.0
        self._emitted = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def sampling_plan(self) -> list[int]:
        """本次打开时被采样的原始帧编号列表。"""
        return list(self._plan)

    def _do_open(self) -> None:
        cv2 = require_cv2()
        if not self._path.is_file():
            msg = f"视频文件不存在：{self._path}"
            raise FrameSourceError(msg)
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            msg = f"视频无法打开（编码不支持或文件损坏）：{self._path}"
            raise FrameSourceError(msg)

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0:
            capture.release()
            msg = f"视频素材为空（帧数为 {total_frames}）：{self._path}"
            raise FrameSourceError(msg)

        self._plan = plan_sampling(
            total_frames, source_fps=self._source_fps, target_fps=self._target_fps
        )
        if self._start is None:
            duration = total_frames / self._source_fps if self._source_fps > 0 else 0.0
            # 末帧锚定到打开时刻，保证离线素材进入默认时间窗口
            self._start = self._clock() - timedelta(seconds=duration)
        self._capture = capture
        self._plan_position = 0
        self._next_raw_index = 0
        self._emitted = 0
        logger.info(
            "视频打开 path=%s total=%s fps=%.3f 采样帧数=%s",
            self._path,
            total_frames,
            self._source_fps,
            len(self._plan),
        )

    def _do_read(self) -> Frame | None:
        assert self._capture is not None  # noqa: S101 - open() 已保证
        assert self._start is not None  # noqa: S101 - open() 已保证

        while self._plan_position < len(self._plan):
            target_index = self._plan[self._plan_position]
            # 顺序读取并跳过未被采样的帧，避免 seek 在不同编码下的兼容性问题
            while self._next_raw_index < target_index:
                if not self._capture.grab():
                    return self._end_of_stream()
                self._next_raw_index += 1
            ok, image = self._capture.read()
            self._next_raw_index += 1
            if not ok or image is None:
                return self._end_of_stream()
            self._plan_position += 1
            self._emitted += 1

            offset = (
                target_index / self._source_fps
                if self._source_fps > 0
                else target_index / self._target_fps
            )
            return Frame(
                frame_id=f"{self.source_id}-{target_index:05d}",
                source_id=self.source_id,
                captured_at=self._start + timedelta(seconds=offset),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                data=image,
                metadata={
                    "kind": "video",
                    "index": target_index,
                    "sampled": self._emitted - 1,
                    "source_fps": self._source_fps,
                    "path": str(self._path),
                },
            )
        return None

    def _end_of_stream(self) -> Frame | None:
        """流提前结束：一帧都没读出来视为读取失败，否则按降级收尾。"""
        if self._emitted == 0:
            msg = f"视频读取失败（无法解码出任何帧）：{self._path}"
            raise FrameSourceError(msg)
        logger.warning(
            "视频提前结束 path=%s 已输出 %s/%s 帧", self._path, self._emitted, len(self._plan)
        )
        self._plan_position = len(self._plan)
        return None

    def _do_close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
