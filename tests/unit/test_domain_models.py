"""领域模型单元测试：分型、有效期与不可变性。"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from agent_visual_context.domain import (
    BBox,
    EpistemicStatus,
    Event,
    Observation,
    PersonCountQuality,
    PersonSceneSummary,
    TrackRef,
)
from tests.conftest import T0


def make_observation(**overrides: object) -> Observation:
    payload: dict[str, object] = {
        "scene_id": "scene-test",
        "subject": TrackRef(track_id="person-01", label="person"),
        "predicate": "looking_at",
        "target": TrackRef(track_id="screen-01", label="screen"),
        "confidence": 0.82,
        "observed_at": T0,
        "expires_at": T0 + timedelta(seconds=4),
        "source_id": "mock-source",
        "model_version": "mock-relate-0.1",
    }
    payload.update(overrides)
    return Observation.model_validate(payload)


def test_observation_carries_required_provenance() -> None:
    observation = make_observation()

    assert observation.epistemic_status is EpistemicStatus.VISUAL_OBSERVATION
    assert observation.source_id == "mock-source"
    assert observation.model_version == "mock-relate-0.1"
    assert observation.ttl_seconds(T0) == pytest.approx(4.0)


def test_observation_expiry_uses_expires_at() -> None:
    observation = make_observation()

    assert not observation.is_expired(T0 + timedelta(seconds=3))
    assert observation.is_expired(T0 + timedelta(seconds=4))


def test_visual_observation_and_rule_event_are_distinct_types() -> None:
    observation = make_observation()
    event = Event(
        scene_id="scene-test",
        kind="greeting-candidate",
        observed_at=T0,
        expires_at=T0 + timedelta(seconds=30),
        payload={"track_id": "person-01"},
    )

    assert observation.epistemic_status is EpistemicStatus.VISUAL_OBSERVATION
    assert event.epistemic_status is EpistemicStatus.RULE_EVENT
    assert not isinstance(observation, Event)


def test_models_are_immutable() -> None:
    observation = make_observation()

    with pytest.raises(ValidationError):
        observation.confidence = 0.1


def test_bbox_rejects_non_positive_size() -> None:
    with pytest.raises(ValidationError):
        BBox(x=0, y=0, width=0, height=10)


def test_person_summary_validates_count_confidence_and_expiry() -> None:
    summary = PersonSceneSummary(
        scene_id="scene-test",
        source_id="camera",
        sampled_at=T0,
        expires_at=T0 + timedelta(seconds=4),
        current_person_count=2,
        confidence=0.8,
        quality=PersonCountQuality.DETECTED,
    )
    assert summary.recognizable_face_count is None
    assert summary.window_distinct_person_count is None
    assert not summary.is_expired(T0)
    assert summary.is_expired(summary.expires_at)

    with pytest.raises(ValidationError):
        PersonSceneSummary(
            scene_id="scene-test",
            source_id="camera",
            sampled_at=T0,
            expires_at=T0 + timedelta(seconds=4),
            current_person_count=-1,
            confidence=1.1,
            quality=PersonCountQuality.DETECTED,
        )
