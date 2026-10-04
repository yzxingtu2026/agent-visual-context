"""有界时间线：所有观察与事件都必须带时间戳与有效期，超期或超量即被裁剪。

时间线是 Agent 可查询的视觉事实来源；它不接受无界增长，也不接受无有效期的条目。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

from ..domain import Event, Observation, utc_now

TimelineItem = Observation | Event


class BoundedTimeline:
    """按容量与 TTL 双重约束的观察/事件时间线。"""

    def __init__(
        self,
        *,
        capacity: int = 512,
        default_ttl: timedelta = timedelta(seconds=4),
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if capacity < 1:
            msg = "capacity 必须大于等于 1"
            raise ValueError(msg)
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._clock = clock
        self._observations: deque[Observation] = deque(maxlen=capacity)
        self._events: deque[Event] = deque(maxlen=capacity)
        self.dropped_total = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def now(self) -> datetime:
        return self._clock()

    def add_observation(
        self, observation: Observation, *, ttl: timedelta | None = None
    ) -> Observation:
        """写入观察。

        - 显式给出 `ttl` 时按 TTL 重算有效期；
        - 未给出时沿用观察自带的 `expires_at`，但不得超过 `default_ttl` 上限，
          以保证时间线内不存在无限期条目（模型不可变，返回补齐后的副本）。
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        limit = observation.observed_at + effective_ttl
        if ttl is not None or observation.expires_at > limit:
            observation = observation.model_copy(update={"expires_at": limit})
        self._append(self._observations, observation)
        return observation

    def add_event(self, event: Event) -> Event:
        self._append(self._events, event)
        return event

    def prune(self, *, now: datetime | None = None) -> int:
        """移除已过期条目，返回移除数量。"""
        moment = now if now is not None else self.now()
        removed = _prune_store(self._observations, moment) + _prune_store(self._events, moment)
        self.dropped_total += removed
        return removed

    def observations(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        now: datetime | None = None,
    ) -> list[Observation]:
        return _select(self._observations, start=start, end=end, now=now)

    def events(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        now: datetime | None = None,
    ) -> list[Event]:
        return _select(self._events, start=start, end=end, now=now)

    def __len__(self) -> int:
        return len(self._observations) + len(self._events)

    def _append[T: (Observation, Event)](self, store: deque[T], item: T) -> None:
        if store.maxlen is not None and len(store) == store.maxlen:
            self.dropped_total += 1
        store.append(item)


def _prune_store[T: (Observation, Event)](store: deque[T], now: datetime) -> int:
    kept = [item for item in store if item.expires_at > now]
    removed = len(store) - len(kept)
    if removed:
        store.clear()
        store.extend(kept)
    return removed


def _select[T: (Observation, Event)](
    items: Iterable[T],
    *,
    start: datetime | None,
    end: datetime | None,
    now: datetime | None,
) -> list[T]:
    result: list[T] = []
    for item in items:
        if start is not None and item.observed_at < start:
            continue
        if end is not None and item.observed_at > end:
            continue
        if now is not None and item.expires_at <= now:
            continue
        result.append(item)
    return result
