"""输入源抽象。

任何图片、视频、摄像头或合成数据源都通过 `FrameSource` 协议接入，
领域层与流水线不感知底层是 OpenCV、文件还是测试替身。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Protocol

from ..domain import Frame
from ..errors import FrameSourceError


class FrameSource(Protocol):
    """帧输入源的结构性接口。

    `read()` 返回 `None` 表示数据源已耗尽；实现方必须保证 `close()` 幂等。
    """

    @property
    def source_id(self) -> str: ...

    def open(self) -> None: ...

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class AbstractFrameSource(ABC):
    """输入源公共基类：统一生命周期、重复打开保护与上下文管理。"""

    def __init__(self, source_id: str) -> None:
        self._source_id = source_id
        self._opened = False

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def is_open(self) -> bool:
        return self._opened

    def open(self) -> None:
        if self._opened:
            return
        self._do_open()
        self._opened = True

    def read(self) -> Frame | None:
        if not self._opened:
            msg = f"输入源 {self._source_id} 未打开，无法读取帧"
            raise FrameSourceError(msg)
        return self._do_read()

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self._do_close()
        finally:
            self._opened = False

    def __enter__(self) -> AbstractFrameSource:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @abstractmethod
    def _do_open(self) -> None: ...

    @abstractmethod
    def _do_read(self) -> Frame | None: ...

    @abstractmethod
    def _do_close(self) -> None: ...
