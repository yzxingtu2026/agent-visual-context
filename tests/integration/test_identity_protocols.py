from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agent_visual_context.api.queries import VisualContextApi
from agent_visual_context.config import AppConfig
from agent_visual_context.domain import FaceObservation, IdentityMatchStatus, PersonIdentityMatch
from agent_visual_context.input import synthetic_frames
from agent_visual_context.perception import (
    IdentityCandidate,
    IdentityMatcher,
    IdentityStore,
    StoredIdentityMatcher,
)
from agent_visual_context.runtime import build_mock_pipeline
from tests.conftest import T0, FakeClock
from tests.integration.test_person_identity_pipeline import FixedFaceAnalyzer
from tests.unit.test_person_identity import face


class SharedStore:
    """模拟宿主的持久化索引；新实例读同一后端数据。"""

    def __init__(self, rows: dict[str, IdentityCandidate]) -> None:
        self.rows = rows
        self.search_calls = 0

    def register(
        self, embedding: Iterable[float], *, model_version: str, now: datetime,
        consent: bool = False, person_id: str | None = None,
    ) -> str:
        if not consent:
            raise PermissionError("consent required")
        del embedding
        resolved_id = person_id or "person-1"
        self.rows[resolved_id] = IdentityCandidate(
            resolved_id, 0.95, model_version, now + timedelta(minutes=1)
        )
        return resolved_id

    def add_sample(
        self, person_id: str, embedding: Iterable[float], *, model_version: str,
        now: datetime, consent: bool = False,
    ) -> None:
        if not consent or person_id not in self.rows or self.rows[person_id].model_version != model_version:
            raise PermissionError("invalid update")
        del embedding, now

    def delete(self, person_id: str) -> None:
        self.rows.pop(person_id, None)

    def disable(self, person_id: str) -> None:
        self.delete(person_id)

    def expire(self, *, now: datetime) -> int:
        expired = [key for key, value in self.rows.items() if value.expires_at is not None
                   and value.expires_at <= now]
        for key in expired:
            self.delete(key)
        return len(expired)

    def search(
        self, embedding: Sequence[float], *, model_version: str, now: datetime,
        limit: int = 2,
    ) -> Sequence[IdentityCandidate]:
        self.search_calls += 1
        del embedding
        return tuple(value for value in self.rows.values()
                     if value.model_version == model_version
                     and (value.expires_at is None or value.expires_at > now))[:limit]


class CustomMatcher:
    def match_faces(
        self, faces: Iterable[FaceObservation], *, source_id: str,
        observed_at: datetime, ttl_seconds: float,
    ) -> tuple[PersonIdentityMatch, ...]:
        return tuple(PersonIdentityMatch(
            track_id=item.track_id or "unknown", person_id="external-person",
            status=IdentityMatchStatus.MATCHED, score=0.91,
            model_version=item.model_version, source_id=source_id,
            observed_at=observed_at,
            expires_at=observed_at + timedelta(seconds=ttl_seconds),
        ) for item in faces if item.track_id is not None)


def test_external_matcher_reaches_pipeline_without_store() -> None:
    clock = FakeClock(T0 + timedelta(seconds=2))
    matcher = CustomMatcher()
    assert isinstance(matcher, IdentityMatcher)
    pipeline = build_mock_pipeline(
        AppConfig(max_frames=1), clock=clock, face_analyzer=FixedFaceAnalyzer(),
        identity_matcher=matcher,
    )
    assert pipeline.identity_store is None
    result = pipeline.run()
    assert result.snapshot is not None and result.snapshot.persons is not None
    assert result.snapshot.persons.identity_matches[0].person_id == "external-person"
    assert "embedding" not in result.snapshot.model_dump_json()


def test_external_store_rebuild_and_rejection_rules() -> None:
    rows: dict[str, IdentityCandidate] = {}
    first_store = SharedStore(rows)
    assert isinstance(first_store, IdentityStore)
    person_id = first_store.register((1, 0), model_version="model-v1", now=T0, consent=True)
    store = SharedStore(rows)
    matcher = StoredIdentityMatcher(store, confirmation_frames=2)
    first = matcher.match_faces([face((1, 0))], source_id="cam", observed_at=T0, ttl_seconds=4)
    second = matcher.match_faces([face((1, 0))], source_id="cam", observed_at=T0 + timedelta(seconds=1), ttl_seconds=4)
    assert first[0].status is IdentityMatchStatus.CANDIDATE
    assert second[0].status is IdentityMatchStatus.MATCHED
    assert second[0].person_id == person_id
    assert store.search_calls == 2
    assert matcher.match_faces([face((1, 0), model="model-v2")], source_id="cam", observed_at=T0, ttl_seconds=4)[0].status is IdentityMatchStatus.UNKNOWN

    rows["person-2"] = IdentityCandidate("person-2", 0.94, "model-v1")
    assert matcher.match_faces([face((1, 0))], source_id="cam", observed_at=T0 + timedelta(seconds=2), ttl_seconds=4)[0].status is IdentityMatchStatus.UNKNOWN
    store.delete("person-2")
    conflicts = matcher.match_faces(
        [face((1, 0), track_id="a"), face((1, 0), track_id="b")],
        source_id="cam", observed_at=T0 + timedelta(seconds=3), ttl_seconds=4,
    )
    assert all(item.status is IdentityMatchStatus.UNKNOWN for item in conflicts)
    store.delete(person_id)
    assert matcher.match_faces([face((1, 0))], source_id="cam", observed_at=T0 + timedelta(seconds=4), ttl_seconds=4)[0].status is IdentityMatchStatus.UNKNOWN


def test_external_store_rebuilds_after_another_process(tmp_path: Path) -> None:
    data_path = tmp_path / "identity-index.json"
    writer = """import json, sys
from datetime import datetime, timedelta
from tests.conftest import T0
from tests.integration.test_identity_protocols import SharedStore
rows = {}
SharedStore(rows).register((1, 0), model_version='model-v1', now=T0,
                           consent=True, person_id='stable-person')
with open(sys.argv[1], 'w') as output:
    json.dump({key: {'score': value.score, 'model_version': value.model_version,
                     'expires_at': value.expires_at.isoformat() if value.expires_at else None}
               for key, value in rows.items()}, output)
"""
    subprocess.run([sys.executable, "-c", writer, str(data_path)], check=True)
    saved = json.loads(data_path.read_text())
    rows = {person_id: IdentityCandidate(
        person_id, item["score"], item["model_version"],
        datetime.fromisoformat(item["expires_at"]) if item["expires_at"] else None,
    ) for person_id, item in saved.items()}
    matcher = StoredIdentityMatcher(SharedStore(rows), confirmation_frames=1)
    result = matcher.match_faces([face((1, 0))], source_id="cam", observed_at=T0, ttl_seconds=4)
    assert result[0].person_id == "stable-person"
    assert result[0].status is IdentityMatchStatus.MATCHED


def test_api_uses_external_store_and_invalidates_confirmation() -> None:
    clock = FakeClock(T0)
    rows: dict[str, IdentityCandidate] = {}
    store = SharedStore(rows)
    pipeline = build_mock_pipeline(AppConfig(max_frames=1), clock=clock, identity_store=store)
    api = VisualContextApi.from_pipeline(pipeline)
    with pytest.raises(PermissionError):
        api.register_anonymous_person((1, 0), model_version="model-v1")
    person_id = api.register_anonymous_person((1, 0), model_version="model-v1", consent=True)
    api.add_anonymous_person_sample(person_id, (1, 0), model_version="model-v1", consent=True)
    api.disable_anonymous_person(person_id)
    assert person_id not in rows


def test_low_quality_does_not_query_store() -> None:
    store = SharedStore({"person-1": IdentityCandidate("person-1", 0.99, "model-v1")})
    matcher = StoredIdentityMatcher(store, min_quality=0.8)
    weak_face = face((1, 0)).model_copy(update={"quality": 0.4})
    result = matcher.match_faces([weak_face], source_id="cam", observed_at=T0, ttl_seconds=4)
    assert result[0].status is IdentityMatchStatus.UNKNOWN
    assert store.search_calls == 0


def test_confirmation_requires_consecutive_face_frames() -> None:
    store = SharedStore({"person-1": IdentityCandidate("person-1", 0.99, "model-v1")})
    matcher = StoredIdentityMatcher(store, confirmation_frames=2)
    first = matcher.match_faces([face((1, 0))], source_id="cam", observed_at=T0, ttl_seconds=4)
    matcher.reset()  # The pipeline calls this when face analysis fails.
    after_gap = matcher.match_faces(
        [face((1, 0))], source_id="cam", observed_at=T0 + timedelta(seconds=1), ttl_seconds=4
    )
    assert first[0].status is IdentityMatchStatus.CANDIDATE
    assert after_gap[0].status is IdentityMatchStatus.CANDIDATE


def test_matcher_error_and_frame_result_do_not_expose_embedding(caplog: pytest.LogCaptureFixture) -> None:
    class FailingMatcher:
        def match_faces(self, faces: Iterable[FaceObservation], *, source_id: str,
                        observed_at: datetime, ttl_seconds: float) -> tuple[PersonIdentityMatch, ...]:
            del faces, source_id, observed_at, ttl_seconds
            raise RuntimeError("private-embedding-123")

    pipeline = build_mock_pipeline(
        AppConfig(max_frames=1), face_analyzer=FixedFaceAnalyzer(),
        identity_matcher=FailingMatcher(),
    )
    frame = synthetic_frames(1, start=T0)[0]
    result = pipeline.process_frame(frame)
    assert "identity_matcher" in result.degraded_components
    assert result.face_items and result.face_items[0].embedding is None
    assert "private-embedding-123" not in caplog.text
    assert "private-embedding-123" not in str(pipeline.status)
