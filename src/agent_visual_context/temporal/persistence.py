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
                stable.append(track.latest)

        self._forget_missing(seen_keys, now=now)
        return stable

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

    def _forget_missing(self, seen_keys: set[tuple[str, str, str]], *, now: datetime) -> None:
        stale_keys = [
            key
            for key, track in self._tracks.items()
            if key not in seen_keys and self._is_stale(track, now)
        ]
        for key in stale_keys:
            del self._tracks[key]
