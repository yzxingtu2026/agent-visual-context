"""策略边界。

策略只产出**候选事件**，不执行任何业务动作（播报、下单、呼叫服务等）。
业务动作必须由 Agent 侧结合用户明确表达与后端权威校验决定。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..domain import Event, Snapshot


class Policy(Protocol):
    """规则策略接口。"""

    @property
    def name(self) -> str: ...

    def evaluate(self, snapshot: Snapshot, *, now: datetime) -> list[Event]: ...
