from __future__ import annotations

from datetime import timedelta

import pytest

from agent_visual_context.domain import BBox, FaceObservation, IdentityMatchStatus
from agent_visual_context.errors import PerceptionError
from agent_visual_context.perception import InMemoryIdentityStore
from tests.conftest import T0


def face(vector: tuple[float, ...], *, track_id: str = "track-1", model: str = "model-v1") -> FaceObservation:
    return FaceObservation(
        bbox=BBox(x=10, y=10, width=40, height=40),
        quality=0.9,
        track_id=track_id,
        model_version=model,
        embedding=vector,
    )


def test_store_is_disabled_by_default_and_requires_consent() -> None:
    store = InMemoryIdentityStore()
    with pytest.raises(PerceptionError):
        store.register((1.0, 0.0), model_version="model-v1", now=T0, consent=True)

    store = InMemoryIdentityStore(enabled=True)
    with pytest.raises(PerceptionError):
        store.register((1.0, 0.0), model_version="model-v1", now=T0)


def test_match_requires_multiple_frames_and_model_version_isolated() -> None:
    store = InMemoryIdentityStore(enabled=True, confirmation_frames=2)
    person_id = store.register((1.0, 0.0), model_version="model-v1", now=T0, consent=True)

    first = store.match_faces(
        [face((1.0, 0.0))], source_id="camera", observed_at=T0, ttl_seconds=4
    )[0]
    second = store.match_faces(
        [face((1.0, 0.0))], source_id="camera", observed_at=T0 + timedelta(seconds=1), ttl_seconds=4
    )[0]
    other_model = store.match_faces(
        [face((1.0, 0.0), model="model-v2")],
        source_id="camera",
        observed_at=T0,
        ttl_seconds=4,
    )[0]

    assert first.status is IdentityMatchStatus.CANDIDATE
    assert second.status is IdentityMatchStatus.MATCHED
    assert second.person_id == person_id
    assert other_model.status is IdentityMatchStatus.UNKNOWN


def test_ambiguous_and_same_frame_conflicting_tracks_are_rejected() -> None:
    store = InMemoryIdentityStore(enabled=True, confirmation_frames=1, ambiguity_margin=0.05)
    first_id = store.register((1.0, 0.0), model_version="model-v1", now=T0, consent=True)
    second_id = store.register((0.99, 0.1), model_version="model-v1", now=T0, consent=True)

    ambiguous = store.match_faces(
        [face((1.0, 0.0))], source_id="camera", observed_at=T0, ttl_seconds=4
    )[0]
    assert ambiguous.status is IdentityMatchStatus.UNKNOWN
    assert {first_id, second_id}.issubset(set(store.person_ids))

    # Two tracks competing for one record are never automatically merged.
    store.delete(second_id)
    conflicts = store.match_faces(
        [face((1.0, 0.0), track_id="track-a"), face((1.0, 0.0), track_id="track-b")],
        source_id="camera",
        observed_at=T0 + timedelta(seconds=1),
        ttl_seconds=4,
    )
    assert all(item.status is IdentityMatchStatus.UNKNOWN for item in conflicts)


def test_delete_and_expire_remove_records() -> None:
    store = InMemoryIdentityStore(enabled=True, retention_seconds=2, confirmation_frames=1)
    person_id = store.register((1.0, 0.0), model_version="model-v1", now=T0, consent=True)
    store.expire(now=T0 + timedelta(seconds=3))
    assert person_id not in store.person_ids

    person_id = store.register((1.0, 0.0), model_version="model-v1", now=T0, consent=True)
    store.delete(person_id)
    result = store.match_faces(
        [face((1.0, 0.0))], source_id="camera", observed_at=T0, ttl_seconds=4
    )[0]
    assert result.status is IdentityMatchStatus.UNKNOWN
