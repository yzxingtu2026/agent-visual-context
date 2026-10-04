"""图片输入源：单张图片或图片序列 -> 统一 `Frame`。

时间戳与编号约定（与视频源共用同一套语义）：
- 帧编号即序列下标，`frame_id = "{source_id}-{index:05d}"`；
- 相邻帧间隔为 `1 / target_fps`；
- 未显式给定 `start` 时，以“最后一帧即打开时刻”为锚点回放，
  保证离线素材落在时间线窗口内（与 `runtime.factory` 的合成帧一致）。

解码依赖 OpenCV，但仅在 `open()` 时惰性加载；像素数据以 numpy 数组
放入 `Frame.data`（`RawFrameData` 对领域层透明），领域层不感知 OpenCV 类型。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

from ..domain import Frame, utc_now
from ..errors import FrameSourceError
from ._opencv import require_cv2
from .base import AbstractFrameSource


class ImageFrameSource(AbstractFrameSource):
    """按顺序回放一组图片文件的输入源；单张图片即长度为 1 的序列。"""

    def __init__(
        self,
        paths: Sequence[Path | str],
        *,
        source_id: str | None = None,
        target_fps: float = 2.0,
        start: datetime | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        resolved = [Path(item) for item in paths]
        if not resolved:
            msg = "图片素材为空：至少需要提供一张图片"
            raise FrameSourceError(msg)
        if target_fps <= 0:
            msg = f"target_fps 必须为正数，收到 {target_fps}"
            raise FrameSourceError(msg)
        resolved_id = source_id if source_id is not None else resolved[0].stem
        super().__init__(resolved_id)
        self._paths = resolved
        self._target_fps = target_fps
        self._start = start
        self._clock = clock
        self._cursor = 0
        self._cv2: ModuleType | None = None

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def _do_open(self) -> None:
        self._cv2 = require_cv2()
        missing = [str(path) for path in self._paths if not path.is_file()]
        if missing:
            msg = f"图片文件不存在：{', '.join(missing)}"
            raise FrameSourceError(msg)
        if self._start is None:
            interval = timedelta(seconds=1 / self._target_fps)
            # 末帧锚定到打开时刻，保证离线素材进入默认时间窗口
            self._start = self._clock() - interval * (len(self._paths) - 1)

    def _do_read(self) -> Frame | None:
        if self._cursor >= len(self._paths):
            return None
        path = self._paths[self._cursor]
        index = self._cursor
        self._cursor += 1

        assert self._cv2 is not None  # noqa: S101 - open() 已保证
        assert self._start is not None  # noqa: S101 - open() 已保证
        image = self._cv2.imread(str(path))
        if image is None:
            msg = f"图片解码失败（文件损坏或格式不支持）：{path}"
            raise FrameSourceError(msg)

        captured_at = self._start + timedelta(seconds=index / self._target_fps)
        return Frame(
            frame_id=f"{self.source_id}-{index:05d}",
            source_id=self.source_id,
            captured_at=captured_at,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            data=image,
            metadata={"kind": "image", "index": index, "path": str(path)},
        )

    def _do_close(self) -> None:
        self._cv2 = None
