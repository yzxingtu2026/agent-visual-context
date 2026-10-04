"""有界帧缓冲：解耦高帧率采集与低频推理（PoC-5）。

实时链路的核心约束是**永不无限堆积**：摄像头以设备原生帧率产出，而关系推理只做低频
（1~2 FPS）处理。`LatestFrameBuffer` 是一个线程安全、容量有界、"丢旧保最新"的缓冲：

- 采集线程持续 `put(frame)`，缓冲满时丢弃最旧帧并累计 `dropped_total`；
- 消费线程用 `get(timeout)` 阻塞等待新帧，取走**最新**一帧并清空更早的积压；
- 缓冲容量默认为 1（只保最新帧），确保推理看到的永远是尽可能新鲜的画面，
  而不是排队积压的历史帧。

本模块只依赖标准库与领域 `Frame`，不感知摄像头、模型或线程编排细节，
供 `runtime/live.py` 的采集/消费线程共享。
"""

from __future__ import annotations

import threading
from collections import deque

from ..domain import Frame


class LatestFrameBuffer:
    """线程安全的有界"保最新帧"缓冲。"""

    def __init__(self, maxsize: int = 1) -> None:
        self._maxsize = max(1, maxsize)
        self._frames: deque[Frame] = deque(maxlen=self._maxsize)
        self._condition = threading.Condition()
        self._dropped_total = 0
        self._put_total = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def dropped_total(self) -> int:
        """因缓冲满而被丢弃的帧总数。"""
        with self._condition:
            return self._dropped_total

    @property
    def put_total(self) -> int:
        """成功写入缓冲的帧总数（含随后被丢弃的）。"""
        with self._condition:
            return self._put_total

    def __len__(self) -> int:
        with self._condition:
            return len(self._frames)

    def put(self, frame: Frame) -> None:
        """写入一帧；缓冲满时丢弃最旧帧以保最新，并唤醒等待的消费者。"""
        with self._condition:
            if len(self._frames) == self._maxsize:
                # deque(maxlen) 会在 append 时自动挤掉最旧一帧
                self._dropped_total += 1
            self._frames.append(frame)
            self._put_total += 1
            self._condition.notify_all()

    def latest(self) -> Frame | None:
        """非阻塞地取走最新帧并清空更早的积压；无帧时返回 `None`。"""
        with self._condition:
            return self._take_latest_locked()

    def get(self, timeout: float | None = None) -> Frame | None:
        """阻塞等待新帧，取走最新帧；超时仍无帧返回 `None`。

        `timeout=None` 表示一直等待直到有帧或缓冲被外部关闭（`get` 本身不阻塞于关闭，
        调用方通过有限超时轮询配合停止信号退出）。
        """
        with self._condition:
            if not self._frames:
                self._condition.wait(timeout)
            return self._take_latest_locked()

    def _take_latest_locked(self) -> Frame | None:
        if not self._frames:
            return None
        frame = self._frames[-1]
        # 实时语义：只处理最新帧，比它更旧的积压帧一律丢弃并计数
        stale = len(self._frames) - 1
        if stale > 0:
            self._dropped_total += stale
        self._frames.clear()
        return frame
