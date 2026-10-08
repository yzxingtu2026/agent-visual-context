"""受控的本地匿名人物特征匹配。

特征库默认关闭。启用后也只保存授权登记的归一化向量，匹配严格按模型版本隔离，
并在单进程内提供过期、禁用和删除能力。向量不会从此模块返回到快照或日志。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..domain import FaceObservation, IdentityMatchStatus, PersonIdentityMatch, new_id
from ..errors import PerceptionError


class IdentityStoreDisabled(PerceptionError):
    """特征库未显式启用。"""


class IdentityAuthorizationError(PerceptionError):
    """登记或更新缺少明确授权。"""


@dataclass(slots=True)
class _FeatureRecord:
    person_id: str
    model_version: str
    samples: list[tuple[float, ...]] = field(default_factory=list)
    expires_at: datetime | None = None
    enabled: bool = True


class InMemoryIdentityStore:
    """默认不持久化的受控特征库。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        match_threshold: float = 0.72,
        candidate_threshold: float = 0.55,
        ambiguity_margin: float = 0.05,
        confirmation_frames: int = 2,
        retention_seconds: float = 86_400.0,
    ) -> None:
        if not 0 <= candidate_threshold <= match_threshold <= 1:
            raise ValueError("candidate_threshold 必须不高于 match_threshold，且均在 0 到 1 之间")
        if ambiguity_margin < 0 or confirmation_frames < 1 or retention_seconds <= 0:
            raise ValueError("歧义边界、确认帧数和保留期必须为有效正数")
        self.enabled = enabled
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        self.ambiguity_margin = ambiguity_margin
        self.confirmation_frames = confirmation_frames
        self.retention_seconds = retention_seconds
        self._records: dict[str, _FeatureRecord] = {}
        self._confirmations: dict[tuple[str, str, str], int] = {}

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise IdentityStoreDisabled("匿名人物特征库未启用")

    @staticmethod
    def _vector(values: Iterable[float]) -> tuple[float, ...]:
        vector = tuple(float(value) for value in values)
        norm = math.sqrt(sum(value * value for value in vector))
        if not vector or norm == 0 or not math.isfinite(norm):
            raise ValueError("特征向量必须为非空有限向量")
        return tuple(value / norm for value in vector)

    def register(
        self,
        embedding: Iterable[float],
        *,
        model_version: str,
        now: datetime,
        consent: bool = False,
        person_id: str | None = None,
    ) -> str:
        """登记匿名人物；调用方必须显式传入授权。"""
        self._require_enabled()
        if not consent:
            raise IdentityAuthorizationError("登记匿名人物特征需要明确授权")
        resolved_id = person_id or new_id("person")
        if resolved_id in self._records:
            raise ValueError(f"person_id 已存在：{resolved_id}")
        self._records[resolved_id] = _FeatureRecord(
            person_id=resolved_id,
            model_version=model_version,
            samples=[self._vector(embedding)],
            expires_at=now + timedelta(seconds=self.retention_seconds),
        )
        return resolved_id

    def add_sample(
        self,
        person_id: str,
        embedding: Iterable[float],
        *,
        model_version: str,
        now: datetime,
        consent: bool = False,
    ) -> None:
        self._require_enabled()
        if not consent:
            raise IdentityAuthorizationError("更新匿名人物特征需要明确授权")
        record = self._records.get(person_id)
        if record is None or not record.enabled:
            raise KeyError(f"未知或已禁用的 person_id：{person_id}")
        if record.model_version != model_version:
            raise ValueError("不同模型版本的特征不能混用；请重新登记")
        record.samples.append(self._vector(embedding))
        record.expires_at = now + timedelta(seconds=self.retention_seconds)

    def delete(self, person_id: str) -> None:
        """删除登记及其全部样本，使其不能再次匹配。"""
        self._records.pop(person_id, None)
        for key in list(self._confirmations):
            if key[1] == person_id:
                del self._confirmations[key]

    def disable(self, person_id: str) -> None:
        record = self._records.get(person_id)
        if record is not None:
            record.enabled = False
        self.delete(person_id)

    def expire(self, *, now: datetime) -> int:
        expired = [
            person_id
            for person_id, record in self._records.items()
            if record.expires_at is not None and record.expires_at <= now
        ]
        for person_id in expired:
            self.delete(person_id)
        return len(expired)

    @property
    def person_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def match_faces(
        self,
        faces: Iterable[FaceObservation],
        *,
        source_id: str,
        observed_at: datetime,
        ttl_seconds: float,
    ) -> tuple[PersonIdentityMatch, ...]:
        """为一帧人脸生成最小匹配结果，冲突轨迹不会自动合并。"""
        face_list = list(faces)
        if not self.enabled:
            return tuple(
                self._unknown(face, source_id=source_id, observed_at=observed_at, ttl_seconds=ttl_seconds)
                for face in face_list
                if face.track_id is not None
            )
        self.expire(now=observed_at)
        proposals: list[tuple[FaceObservation, str | None, float | None, IdentityMatchStatus]] = []
        for face in face_list:
            if face.track_id is None or face.embedding is None:
                continue
            proposals.append((face, *self._propose(face, observed_at)))

        matched_tracks: dict[str, list[int]] = {}
        for index, (_, person_id, _, status) in enumerate(proposals):
            if person_id is not None and status is not IdentityMatchStatus.UNKNOWN:
                matched_tracks.setdefault(person_id, []).append(index)
        conflicts = {index for indexes in matched_tracks.values() if len(indexes) > 1 for index in indexes}

        results: list[PersonIdentityMatch] = []
        for index, (face, person_id, score, status) in enumerate(proposals):
            if index in conflicts:
                person_id = None
                status = IdentityMatchStatus.UNKNOWN
            results.append(
                PersonIdentityMatch(
                    track_id=face.track_id or "unknown-track",
                    person_id=person_id,
                    status=status,
                    score=score,
                    model_version=face.model_version,
                    source_id=source_id,
                    observed_at=observed_at,
                    expires_at=observed_at + timedelta(seconds=ttl_seconds),
                )
            )
        return tuple(results)

    def _propose(
        self, face: FaceObservation, observed_at: datetime
    ) -> tuple[str | None, float | None, IdentityMatchStatus]:
        assert face.track_id is not None
        vector = self._vector(face.embedding or ())
        ranked: list[tuple[float, str]] = []
        for person_id, record in self._records.items():
            if not record.enabled or record.model_version != face.model_version:
                continue
            if record.expires_at is not None and record.expires_at <= observed_at:
                continue
            score = max(sum(a * b for a, b in zip(vector, sample, strict=True)) for sample in record.samples)
            ranked.append((score, person_id))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] < self.candidate_threshold:
            return None, ranked[0][0] if ranked else None, IdentityMatchStatus.UNKNOWN
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.ambiguity_margin:
            return None, ranked[0][0], IdentityMatchStatus.UNKNOWN
        score, person_id = ranked[0]
        if score < self.match_threshold:
            return person_id, score, IdentityMatchStatus.CANDIDATE
        key = (face.track_id, person_id, face.model_version)
        self._confirmations[key] = self._confirmations.get(key, 0) + 1
        status = (
            IdentityMatchStatus.MATCHED
            if self._confirmations[key] >= self.confirmation_frames
            else IdentityMatchStatus.CANDIDATE
        )
        return person_id, score, status

    @staticmethod
    def _unknown(
        face: FaceObservation, *, source_id: str, observed_at: datetime, ttl_seconds: float
    ) -> PersonIdentityMatch:
        return PersonIdentityMatch(
            track_id=face.track_id or "unknown-track",
            status=IdentityMatchStatus.UNKNOWN,
            model_version=face.model_version,
            source_id=source_id,
            observed_at=observed_at,
            expires_at=observed_at + timedelta(seconds=ttl_seconds),
        )
