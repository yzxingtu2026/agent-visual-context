"""匿名身份协议、匹配策略与本地测试存储。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from ..domain import FaceObservation, IdentityMatchStatus, PersonIdentityMatch, new_id
from ..errors import PerceptionError


class IdentityStoreDisabled(PerceptionError):
    """特征库未显式启用。"""


class IdentityAuthorizationError(PerceptionError):
    """登记或更新缺少明确授权。"""


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    """受控检索返回的匿名候选，不含向量。"""

    person_id: str
    score: float
    model_version: str
    expires_at: datetime | None = None


@runtime_checkable
class IdentityStore(Protocol):
    """由宿主实现持久化、授权、删除和按模型版本隔离的向量检索。"""

    def register(
        self, embedding: Iterable[float], *, model_version: str, now: datetime,
        consent: bool = False, person_id: str | None = None,
    ) -> str: ...

    def add_sample(
        self, person_id: str, embedding: Iterable[float], *, model_version: str,
        now: datetime, consent: bool = False,
    ) -> None: ...

    def delete(self, person_id: str) -> None: ...

    def disable(self, person_id: str) -> None: ...

    def expire(self, *, now: datetime) -> int: ...

    def search(
        self, embedding: Sequence[float], *, model_version: str, now: datetime,
        limit: int = 2,
    ) -> Sequence[IdentityCandidate]: ...


@runtime_checkable
class IdentityMatcher(Protocol):
    """单帧匹配边界；调用期间可读取 embedding，只返回最小结果。"""

    def match_faces(
        self, faces: Iterable[FaceObservation], *, source_id: str,
        observed_at: datetime, ttl_seconds: float,
    ) -> tuple[PersonIdentityMatch, ...]: ...


def _vector(values: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or norm == 0 or not math.isfinite(norm):
        raise ValueError("特征向量必须为非空有限向量")
    return tuple(value / norm for value in vector)


class StoredIdentityMatcher:
    """在受控 store 的前两名候选上应用阈值、多帧确认及同帧冲突规则。"""

    def __init__(
        self, store: IdentityStore, *, match_threshold: float = 0.72,
        candidate_threshold: float = 0.55, ambiguity_margin: float = 0.05,
        confirmation_frames: int = 2, min_quality: float = 0.5,
    ) -> None:
        if not 0 <= candidate_threshold <= match_threshold <= 1:
            raise ValueError("candidate_threshold 必须不高于 match_threshold，且均在 0 到 1 之间")
        if ambiguity_margin < 0 or confirmation_frames < 1 or not 0 <= min_quality <= 1:
            raise ValueError("歧义边界、确认帧数和人脸质量必须有效")
        self.store = store
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        self.ambiguity_margin = ambiguity_margin
        self.confirmation_frames = confirmation_frames
        self.min_quality = min_quality
        self._confirmations: dict[tuple[str, str, str, str], tuple[int, datetime]] = {}

    def forget_person(self, person_id: str) -> None:
        self._confirmations = {
            key: count for key, count in self._confirmations.items() if key[2] != person_id
        }

    def reset(self) -> None:
        """上游丢帧或人脸分析失败时中断多帧确认。"""
        self._confirmations.clear()

    def match_faces(
        self, faces: Iterable[FaceObservation], *, source_id: str,
        observed_at: datetime, ttl_seconds: float,
    ) -> tuple[PersonIdentityMatch, ...]:
        proposals: list[tuple[FaceObservation, str | None, float | None, IdentityMatchStatus]] = []
        for face in faces:
            if face.track_id is None:
                continue
            person_id: str | None = None
            score: float | None = None
            status = IdentityMatchStatus.UNKNOWN
            if face.embedding is not None and face.quality >= self.min_quality:
                candidates = list(self.store.search(
                    _vector(face.embedding), model_version=face.model_version,
                    now=observed_at, limit=2,
                ))
                candidates = sorted(
                    (item for item in candidates if item.model_version == face.model_version
                     and (item.expires_at is None or item.expires_at > observed_at)
                     and math.isfinite(item.score) and 0 <= item.score <= 1),
                    key=lambda item: item.score, reverse=True,
                )
                if candidates:
                    score = candidates[0].score
                    if (score >= self.candidate_threshold
                        and (len(candidates) < 2
                             or score - candidates[1].score >= self.ambiguity_margin)):
                        person_id = candidates[0].person_id
                        status = (IdentityMatchStatus.MATCHED if score >= self.match_threshold
                                  else IdentityMatchStatus.CANDIDATE)
            proposals.append((face, person_id, score, status))

        counts: dict[str, int] = {}
        for _, person_id, _, _ in proposals:
            if person_id is not None:
                counts[person_id] = counts.get(person_id, 0) + 1

        results: list[PersonIdentityMatch] = []
        next_confirmations: dict[tuple[str, str, str, str], tuple[int, datetime]] = {}
        for face, person_id, score, status in proposals:
            assert face.track_id is not None
            if person_id is None or counts.get(person_id, 0) > 1:
                status = IdentityMatchStatus.UNKNOWN
                person_id = None
            if status is IdentityMatchStatus.MATCHED and person_id is not None:
                key = (source_id, face.track_id, person_id, face.model_version)
                previous = self._confirmations.get(key)
                count = previous[0] + 1 if previous is not None and previous[1] < observed_at else 1
                next_confirmations[key] = (count, observed_at)
                if count < self.confirmation_frames:
                    status = IdentityMatchStatus.CANDIDATE
            results.append(PersonIdentityMatch(
                track_id=face.track_id, person_id=person_id, status=status, score=score,
                model_version=face.model_version, source_id=source_id,
                observed_at=observed_at, expires_at=observed_at + timedelta(seconds=ttl_seconds),
            ))
        self._confirmations = next_confirmations
        return tuple(results)


@dataclass(slots=True)
class _FeatureRecord:
    person_id: str
    model_version: str
    samples: list[tuple[float, ...]] = field(default_factory=list)
    expires_at: datetime | None = None
    enabled: bool = True


class InMemoryIdentityStore:
    """默认关闭、仅供本地验证的进程内 IdentityStore。"""

    def __init__(
        self, *, enabled: bool = False, match_threshold: float = 0.72,
        candidate_threshold: float = 0.55, ambiguity_margin: float = 0.05,
        confirmation_frames: int = 2, retention_seconds: float = 86_400.0,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("保留期必须为正数")
        self.enabled = enabled
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        self.ambiguity_margin = ambiguity_margin
        self.confirmation_frames = confirmation_frames
        self.retention_seconds = retention_seconds
        self._records: dict[str, _FeatureRecord] = {}
        self._matcher = StoredIdentityMatcher(
            self, match_threshold=match_threshold, candidate_threshold=candidate_threshold,
            ambiguity_margin=ambiguity_margin, confirmation_frames=confirmation_frames,
        )

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise IdentityStoreDisabled("匿名人物特征库未启用")

    def register(
        self, embedding: Iterable[float], *, model_version: str, now: datetime,
        consent: bool = False, person_id: str | None = None,
    ) -> str:
        self._require_enabled()
        if not consent:
            raise IdentityAuthorizationError("登记匿名人物特征需要明确授权")
        resolved_id = person_id or new_id("person")
        if resolved_id in self._records:
            raise ValueError(f"person_id 已存在：{resolved_id}")
        self._records[resolved_id] = _FeatureRecord(
            person_id=resolved_id, model_version=model_version,
            samples=[_vector(embedding)],
            expires_at=now + timedelta(seconds=self.retention_seconds),
        )
        return resolved_id

    def add_sample(
        self, person_id: str, embedding: Iterable[float], *, model_version: str,
        now: datetime, consent: bool = False,
    ) -> None:
        self._require_enabled()
        if not consent:
            raise IdentityAuthorizationError("更新匿名人物特征需要明确授权")
        record = self._records.get(person_id)
        if record is None or not record.enabled or (record.expires_at is not None and record.expires_at <= now):
            raise KeyError(f"未知、已禁用或已过期的 person_id：{person_id}")
        if record.model_version != model_version:
            raise ValueError("不同模型版本的特征不能混用；请重新登记")
        record.samples.append(_vector(embedding))
        record.expires_at = now + timedelta(seconds=self.retention_seconds)

    def delete(self, person_id: str) -> None:
        self._records.pop(person_id, None)
        self._matcher.forget_person(person_id)

    def disable(self, person_id: str) -> None:
        self.delete(person_id)

    def expire(self, *, now: datetime) -> int:
        expired = [person_id for person_id, record in self._records.items()
                   if record.expires_at is not None and record.expires_at <= now]
        for person_id in expired:
            self.delete(person_id)
        return len(expired)

    @property
    def person_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def search(
        self, embedding: Sequence[float], *, model_version: str, now: datetime,
        limit: int = 2,
    ) -> Sequence[IdentityCandidate]:
        if not self.enabled:
            return ()
        if limit < 1:
            raise ValueError("limit 必须为正数")
        vector = _vector(embedding)
        self.expire(now=now)
        ranked = []
        for record in self._records.values():
            if not record.enabled or record.model_version != model_version:
                continue
            scores = [sum(a * b for a, b in zip(vector, sample, strict=True))
                      for sample in record.samples if len(sample) == len(vector)]
            if scores:
                ranked.append(IdentityCandidate(
                    person_id=record.person_id, score=max(0.0, min(1.0, max(scores))),
                    model_version=model_version, expires_at=record.expires_at,
                ))
        return tuple(sorted(ranked, key=lambda item: item.score, reverse=True)[:limit])

    def match_faces(
        self, faces: Iterable[FaceObservation], *, source_id: str,
        observed_at: datetime, ttl_seconds: float,
    ) -> tuple[PersonIdentityMatch, ...]:
        """兼容 #18 的直接调用；新接入方应注入 StoredIdentityMatcher。"""
        return self._matcher.match_faces(
            faces, source_id=source_id, observed_at=observed_at, ttl_seconds=ttl_seconds,
        )
