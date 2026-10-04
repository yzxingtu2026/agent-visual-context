"""跨帧去抖与关系持续性。

单帧关系抖动不应直接进入时间线：只有连续命中且持续时间达标的关系才被提升为观察。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..domain import Relation


@dataclass(slots=True)
class _RelationTrack:
    hits: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    peak_confidence: float = 0.0
    latest: Relation | None = field(default=None)
    # 该关系是否曾经被提升为稳定观察；用于区分“结束后消失”与“从未成形即消失”。
    promoted: bool = False


@dataclass(frozen=True, slots=True)
class GateUpdate:
    """一次去抖更新的完整结果。

    - `stable`：本帧满足持续性条件的稳定关系（可提升为观察）；
    - `ended`：曾经稳定、但已超过最大间隔未再命中的关系（关系“结束”信号），
      返回的是结束前的最后一条证据。
    """

    stable: list[Relation] = field(default_factory=list)
    ended: list[Relation] = field(default_factory=list)


class PersistenceGate:
    """按最小命中次数、最小持续时间与最大间隔过滤关系。"""

    def __init__(
        self,
        *,
        min_hits: int = 2,
        min_seconds: float = 1.0,
        max_gap_seconds: float = 2.0,
    ) -> None:
        if min_hits < 1:
            msg = "min_hits 必须大于等于 1"
            raise ValueError(msg)
        self._min_hits = min_hits
        self._min_seconds = min_seconds
        self._max_gap_seconds = max_gap_seconds
        self._tracks: dict[tuple[str, str, str], _RelationTrack] = {}

    def update(self, relations: list[Relation], *, now: datetime) -> list[Relation]:
        """输入当帧关系，返回已满足持续性条件的关系（去抖后的稳定关系）。"""
        return self.update_with_lifecycle(relations, now=now).stable

    def update_with_lifecycle(self, relations: list[Relation], *, now: datetime) -> GateUpdate:
        """输入当帧关系，返回稳定关系与结束关系。

        关系的完整生命周期由此暴露：
        - **开始/持续**：命中累积到 `min_hits` 且持续时长达标后进入 `stable`；
        - **结束**：曾经稳定的关系超过 `max_gap_seconds` 未再命中，被遗忘并进入 `ended`。
        """
        seen_keys: set[tuple[str, str, str]] = set()
        stable: list[Relation] = []

        for relation in relations:
            key = relation.key
            seen_keys.add(key)
            track = self._tracks.get(key)
            if track is None or self._is_stale(track, now):
                track = _RelationTrack()
                self._tracks[key] = track

            track.hits += 1
            track.first_seen = track.first_seen or now
            track.last_seen = now
            track.peak_confidence = max(track.peak_confidence, relation.confidence)
            # 始终保留最新一条关系：时间戳代表最近一次证据，避免快照窗口把观察过滤掉
            track.latest = relation

            if self._is_stable(track) and track.latest is not None:
                track.promoted = True
                stable.append(track.latest)

        ended = self._forget_missing(seen_keys, now=now)
        return GateUpdate(stable=stable, ended=ended)

    def reset(self) -> None:
        self._tracks.clear()

    def _is_stale(self, track: _RelationTrack, now: datetime) -> bool:
        if track.last_seen is None:
            return False
        return (now - track.last_seen).total_seconds() > self._max_gap_seconds

    def _is_stable(self, track: _RelationTrack) -> bool:
        if track.hits < self._min_hits or track.first_seen is None or track.last_seen is None:
            return False
        return (track.last_seen - track.first_seen).total_seconds() >= self._min_seconds

    def _forget_missing(
        self, seen_keys: set[tuple[str, str, str]], *, now: datetime
    ) -> list[Relation]:
        """遗忘超过最大间隔未再命中的关系，返回其中曾经稳定者的结束证据。"""
        ended: list[Relation] = []
        stale_keys = [
            key
            for key, track in self._tracks.items()
            if key not in seen_keys and self._is_stale(track, now)
        ]
        for key in stale_keys:
            track = self._tracks.pop(key)
            # 只有曾经成形（提升为稳定）的关系才算“结束”；从未成形的抖动直接静默丢弃。
            if track.promoted and track.latest is not None:
                ended.append(track.latest)
        return ended
