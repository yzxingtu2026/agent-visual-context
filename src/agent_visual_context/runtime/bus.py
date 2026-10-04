"""事件总线：进程内的最小发布/订阅边界。

后续接入 WebSocket、Agent 订阅或跨进程通道时，只需替换本模块实现，
流水线与 API 层的调用方式保持不变。
"""

from __future__ import annotations

from collections.abc import Callable

from ..domain import Event
from ..logging_setup import get_logger

EventListener = Callable[[Event], None]

logger = get_logger("runtime.bus")


class EventBus:
    """同步事件总线：订阅者异常被隔离，不影响流水线。"""

    def __init__(self) -> None:
        self._listeners: dict[int, EventListener] = {}

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        key = id(listener)
        self._listeners[key] = listener

        def unsubscribe() -> None:
            self._listeners.pop(key, None)

        return unsubscribe

    def publish(self, event: Event) -> None:
        for listener in list(self._listeners.values()):
            try:
                listener(event)
            except Exception:
                # 订阅者异常必须被隔离，避免影响视觉流水线
                logger.exception("事件订阅者处理失败，已忽略 kind=%s", event.kind)

    @property
    def listener_count(self) -> int:
        return len(self._listeners)
