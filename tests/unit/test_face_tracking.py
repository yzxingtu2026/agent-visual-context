from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import sleep
from types import SimpleNamespace

import pytest

from agent_visual_context.config import AppConfig
from agent_visual_context.domain import BBox, Detection, FaceObservation, Frame
from agent_visual_context.perception.face_association import associate_faces
from agent_visual_context.perception.insightface import InsightFaceAnalyzer
from agent_visual_context.perception.short_tracker import ShortTermTracker
from agent_visual_context.runtime import build_mock_pipeline

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def frame(index: int, *, data: object = None) -> Frame:
    return Frame(
        frame_id=str(index),
        source_id="test",
        captured_at=NOW + timedelta(seconds=index * 0.2),
        width=300,
        height=200,
        data=data,
    )


def person(x: float, at: datetime) -> Detection:
    return Detection(
        label="person", bbox=BBox(x=x, y=20, width=40, height=100), confidence=0.9, detected_at=at
    )


def face(x: float, *, quality: float = 0.9) -> FaceObservation:
    return FaceObservation(
        bbox=BBox(x=x, y=25, width=25, height=35),
        quality=quality,
        model_version="fake",
        age_band="30-39",
        embedding=(0.1, 0.2),
    )


def test_crossing_reorder_and_reentry_do_not_reuse_track_id() -> None:
    tracker = ShortTermTracker(max_gap_seconds=0.5)
    positions = [(0, 100), (20, 80), (40, 60), (60, 40), (80, 20)]
    first_ids: tuple[str, str] | None = None
    for index, (left, right) in enumerate(positions):
        current = frame(index)
        detections = [person(left, current.captured_at), person(right, current.captured_at)]
        if index % 2:
            detections.reverse()
        tracked = tracker.update(current, detections)
        by_x = {item.bbox.x: item.track_id for item in tracked}
        if first_ids is None:
            first_ids = (by_x[left], by_x[right])
        assert by_x[left] == first_ids[0]
        assert by_x[right] == first_ids[1]
    assert tracker.update(frame(5), []) == []
    returning = tracker.update(frame(10), [person(80, frame(10).captured_at)])
    assert first_ids is not None
    assert returning[0].track_id not in first_ids
    tracker.reset()
    assert (
        tracker.update(frame(11), [person(80, frame(11).captured_at)])[0].track_id
        != returning[0].track_id
    )


def test_face_association_rejects_ambiguous_and_low_quality() -> None:
    tracker = ShortTermTracker()
    current = frame(0)
    tracks = tracker.update(
        current, [person(0, current.captured_at), person(100, current.captured_at)]
    )
    matches = associate_faces([face(7), face(107), face(7, quality=0.2)], tracks)
    assert [item.track_id for item in matches] == [tracks[0].track_id, tracks[1].track_id, None]
    overlapping = tracker.update(
        frame(1), [person(0, frame(1).captured_at), person(10, frame(1).captured_at)]
    )
    assert associate_faces([face(15)], overlapping)[0].track_id is None


def test_insightface_backend_is_lazy_and_embedding_is_not_serialized(tmp_path: object) -> None:
    calls: list[object] = []

    class Backend:
        def get(self, image: object) -> list[object]:
            calls.append(image)
            return [
                SimpleNamespace(
                    bbox=[5, 5, 45, 45],
                    det_score=0.95,
                    kps=[[10, 10], [20, 10]],
                    normed_embedding=[0.1, 0.2],
                    age=34,
                    gender=0,
                )
            ]

    analyzer = InsightFaceAnalyzer(model_dir=tmp_path, backend_factory=Backend)  # type: ignore[arg-type]
    assert not calls
    result = analyzer.analyze(frame(0, data=object()))
    analyzer.close()
    assert len(calls) == 1
    assert result[0].age_band == "30-39"
    assert result[0].apparent_gender == "female"
    assert result[0].embedding == (0.1, 0.2)
    assert "embedding" not in result[0].model_dump()


def test_face_failure_keeps_person_count_and_relations() -> None:
    class FailingAnalyzer:
        model_version = "fake"

        def analyze(self, current: Frame) -> list[FaceObservation]:
            raise RuntimeError("unavailable")

    config = AppConfig(max_frames=1)
    pipeline = build_mock_pipeline(config, face_analyzer=FailingAnalyzer())
    result = pipeline.process_frame(frame(0, data=object()))
    assert result.persons is not None
    assert result.persons.current_person_count == 1
    assert result.persons.recognizable_face_count is None
    assert result.relations == 1
    assert result.degraded_components == ("face_analyzer",)
    assert pipeline.status.components["face_analyzer"].failures == 1


def test_pipeline_exposes_face_count_and_estimate_without_vector() -> None:
    class Analyzer:
        model_version = "fake"

        def analyze(self, current: Frame) -> list[FaceObservation]:
            return [face(110).model_copy(update={"bbox": BBox(x=110, y=130, width=25, height=35)})]

    pipeline = build_mock_pipeline(AppConfig(max_frames=1), face_analyzer=Analyzer())
    result = pipeline.process_frame(frame(0, data=object()))
    assert result.persons is not None
    assert result.persons.current_person_count == 1
    assert result.persons.recognizable_face_count == 1
    assert len(result.persons.face_estimates) == 1
    assert result.persons.face_estimates[0].track_id == "person-01"
    assert "embedding" not in result.persons.model_dump_json()


def test_face_timeout_does_not_queue_more_inference(tmp_path: object) -> None:
    calls = 0

    class SlowBackend:
        def get(self, image: object) -> list[object]:
            nonlocal calls
            calls += 1
            sleep(0.05)
            return []

    analyzer = InsightFaceAnalyzer(
        model_dir=tmp_path, timeout_seconds=0.001, backend_factory=SlowBackend
    )  # type: ignore[arg-type]
    with pytest.raises(Exception, match="超过"):
        analyzer.analyze(frame(0, data=object()))
    with pytest.raises(Exception, match="仍在运行"):
        analyzer.analyze(frame(1, data=object()))
    analyzer.close()
    assert calls == 1


def test_missing_local_weights_degrades_without_download(tmp_path: object) -> None:
    analyzer = InsightFaceAnalyzer(model_dir=tmp_path, timeout_seconds=0.1)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="本地权重缺失"):
        analyzer.analyze(frame(0, data=object()))
    analyzer.close()
