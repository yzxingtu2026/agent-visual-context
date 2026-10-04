"""本地摄像头输入源（PoC-5）：实时采集 -> 统一 `Frame`。

与离线图片/视频源共用 `FrameSource` 协议与 `AbstractFrameSource` 生命周期基类，
但语义上摄像头是**无限实时流**：

- **启动/停止**：直接复用基类的 `open()`/`close()`（即启动/停止采集）与上下文管理器，
  不额外发明一套生命周期；`close()` 幂等，进程退出时兜底释放设备。
- **设备选择**：`device_index` 选择摄像头（0 为默认设备），分辨率由 `width`/`height` 请求，
  实际生效尺寸以设备返回为准并写入 `Frame.width/height`。
- **低频采样**：`target_fps` 是发射节流口径——`read()` 保证相邻两帧发射间隔不小于
  `1 / target_fps`，未到期时通过注入的 `sleeper` 等待后再抓取最新帧，避免高频采集空转。
- **时间戳与编号**：`captured_at` 取自 `clock()`（真实采集时刻），`frame_id = "{source_id}-{序号:05d}"`
  按发射顺序递增；摄像头无"源帧号"概念，序号即已发射帧计数。

依赖隔离与降级（与离线源一致）：
- OpenCV 通过 `input/_opencv.py::require_cv2()` 惰性加载，包导入不强制依赖 cv2；
- 真实设备采集走 `OpenCVCameraBackend`，测试与离线验证可注入假后端，无需真实摄像头；
- 设备打不开或读帧失败抛 `FrameSourceError`，由 `runtime` 降级处理（不阻塞宿主链路）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Protocol

from ..domain import Frame, utc_now
from ..errors import FrameSourceError
from ..logging_setup import get_logger
from ._opencv import require_cv2
from .base import AbstractFrameSource

logger = get_logger("input.camera")


class CameraBackend(Protocol):
    """摄像头采集后端协议：真实实现基于 OpenCV，测试可注入假后端。

    后端只负责设备交互（打开/读一帧/释放），把原始图像交给 `CameraSource`，
    不感知节流、编号、时间戳与领域模型。
    """

    def open(self) -> None:
        """打开设备；失败时抛出异常，由 `CameraSource` 转为 `FrameSourceError`。"""
        ...

    def read(self) -> Any | None:
        """读取一帧原始图像；无可用帧时返回 `None`。"""
        ...

    def release(self) -> None:
        """释放设备；必须幂等。"""
        ...


class OpenCVCameraBackend:
    """基于 `cv2.VideoCapture` 的真实摄像头后端。

    构造时不打开设备，`open()` 时才惰性加载 cv2 并申请设备，失败抛 `FrameSourceError`。
    """

    def __init__(self, device_index: int, *, width: int, height: int) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        # cv2.VideoCapture 无公开类型别名，运行时由 OpenCV 提供
        self._capture: Any = None

    def open(self) -> None:
        cv2 = require_cv2()
        capture = cv2.VideoCapture(self._device_index)
        if not capture.isOpened():
            capture.release()
            msg = f"摄像头无法打开（设备不存在或被占用）：index={self._device_index}"
            raise FrameSourceError(msg)
        # 请求分辨率；设备可能不完全支持，实际尺寸以读取到的帧为准。
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture = capture

    def read(self) -> Any | None:
        if self._capture is None:
            return None
        ok, image = self._capture.read()
        return image if ok and image is not None else None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class CameraSource(AbstractFrameSource):
    """从本地摄像头按目标帧率采集帧的实时输入源。

    参数：
    - `device_index`：摄像头设备索引（默认 0）；
    - `width`/`height`：请求分辨率；
    - `target_fps`：发射节流帧率，>0；
    - `backend`：已构造的后端实例（测试注入用），给出后不再惰性创建；
    - `backend_factory`：后端构造函数，默认 `OpenCVCameraBackend`；
    - `sleeper`：节流等待函数，默认 `time.sleep`，测试可注入以推进假时钟；
    - `clock`：时间源，默认 `utc_now`。
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: float = 2.0,
        source_id: str | None = None,
        backend: CameraBackend | None = None,
        backend_factory: Callable[[int, int, int], CameraBackend] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if target_fps <= 0:
            msg = f"target_fps 必须为正数，收到 {target_fps}"
            raise FrameSourceError(msg)
        if device_index < 0:
            msg = f"device_index 不能为负数，收到 {device_index}"
            raise FrameSourceError(msg)
        super().__init__(source_id if source_id is not None else f"camera-{device_index}")
        self._device_index = device_index
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._interval = timedelta(seconds=1 / target_fps)
        self._backend = backend
        self._backend_factory = backend_factory or _default_backend_factory
        self._sleeper = sleeper
        self._clock = clock
        self._emitted = 0
        self._last_captured_at: datetime | None = None
        self._open_failed = False

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def target_fps(self) -> float:
        return self._target_fps

    @property
    def frames_emitted(self) -> int:
        return self._emitted

    @property
    def last_captured_at(self) -> datetime | None:
        return self._last_captured_at

    def _do_open(self) -> None:
        if self._backend is None:
            if self._open_failed:
                msg = "摄像头此前打开失败，已降级；跳过重复申请设备。"
                raise FrameSourceError(msg)
            try:
                self._backend = self._backend_factory(self._device_index, self._width, self._height)
            except Exception as exc:
                self._open_failed = True
                msg = f"摄像头后端初始化失败，已降级：{exc}"
                raise FrameSourceError(msg) from exc
        try:
            self._backend.open()
        except FrameSourceError:
            self._open_failed = True
            raise
        except Exception as exc:
            self._open_failed = True
            msg = f"摄像头打开失败，已降级：{exc}"
            raise FrameSourceError(msg) from exc
        self._emitted = 0
        self._last_captured_at = None
        logger.info(
            "摄像头打开 device=%s 请求分辨率=%sx%s target_fps=%.2f",
            self._device_index,
            self._width,
            self._height,
            self._target_fps,
        )

    def _do_read(self) -> Frame | None:
        assert self._backend is not None  # noqa: S101 - open() 已保证
        self._throttle()
        image = self._backend.read()
        if image is None:
            # 实时流读帧失败按可恢复异常处理，由 runtime 降级为"无帧"。
            msg = f"摄像头读帧失败：device={self._device_index}"
            raise FrameSourceError(msg)

        captured_at = self._clock()
        index = self._emitted
        self._emitted += 1
        self._last_captured_at = captured_at
        return Frame(
            frame_id=f"{self.source_id}-{index:05d}",
            source_id=self.source_id,
            captured_at=captured_at,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            data=image,
            metadata={
                "kind": "camera",
                "index": index,
                "device_index": self._device_index,
                "target_fps": self._target_fps,
            },
        )

    def _throttle(self) -> None:
        """按 `target_fps` 节流：距上次发射不足一个间隔时等待后再抓取最新帧。"""
        if self._last_captured_at is None:
            return
        elapsed = self._clock() - self._last_captured_at
        remaining = self._interval - elapsed
        if remaining > timedelta(0):
            self._sleeper(remaining.total_seconds())

    def _do_close(self) -> None:
        if self._backend is not None:
            self._backend.release()
        logger.info("摄像头关闭 device=%s 已发射 %s 帧", self._device_index, self._emitted)


def _default_backend_factory(device_index: int, width: int, height: int) -> CameraBackend:
    return OpenCVCameraBackend(device_index, width=width, height=height)
