from __future__ import annotations

from agent_visual_context.config import AppConfig
from agent_visual_context.domain import BBox, FaceObservation, IdentityMatchStatus
from agent_visual_context.perception import InMemoryIdentityStore
from agent_visual_context.runtime import build_mock_pipeline
from tests.conftest import FakeClock


class FixedFaceAnalyzer:
    model_version = "model-v1"

    def analyze(self, frame: object) -> list[FaceObservation]:
        del frame
        return [
            FaceObservation(
                bbox=BBox(x=110, y=145, width=40, height=45),
                quality=0.95,
                model_version=self.model_version,
                embedding=(1.0, 0.0),
            )
        ]


def test_identity_match_is_minimal_and_reaches_snapshot(clock: FakeClock) -> None:
    config = AppConfig(
        max_frames=2,
        identity_matching_enabled=True,
        identity_confirmation_frames=2,
        persistence_min_hits=1,
        persistence_min_seconds=0,
    )
    store = InMemoryIdentityStore(enabled=True, confirmation_frames=2)
    person_id = store.register((1.0, 0.0), model_version="model-v1", now=clock(), consent=True)
    pipeline = build_mock_pipeline(
        config,
        clock=clock,
        face_analyzer=FixedFaceAnalyzer(),
        identity_store=store,
    )

    result = pipeline.run()

    assert result.snapshot is not None
    assert result.snapshot.persons is not None
    matches = result.snapshot.persons.identity_matches
    assert len(matches) == 1
    assert matches[0].status is IdentityMatchStatus.MATCHED
    assert matches[0].person_id == person_id
    serialized = result.snapshot.model_dump()
    assert "embedding" not in str(serialized)
    assert "person_id" in serialized["persons"]["identity_matches"][0]
