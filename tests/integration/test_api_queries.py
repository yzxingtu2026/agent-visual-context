"""集成测试：Agent 查询/订阅接口边界。"""

from __future__ import annotations

from datetime import timedelta

from agent_visual_context.api import VisualContextApi
from agent_visual_context.config import AppConfig
from agent_visual_context.domain import Event
from agent_visual_context.runtime import Pipeline, build_mock_pipeline
from tests.conftest import FakeClock


def run_pipeline(config: AppConfig, clock: FakeClock) -> tuple[VisualContextApi, Pipeline]:
    pipeline = build_mock_pipeline(config, clock=clock)
    pipeline.run()
    return VisualContextApi.from_pipeline(pipeline), pipeline


def test_snapshot_falls_back_to_configured_scene_id(config: AppConfig, clock: FakeClock) -> None:
    api, _ = run_pipeline(config, clock)

    snapshot = api.get_scene_snapshot(now=clock())

    assert snapshot.scene_id == config.scene_id
    assert snapshot.highlights


def test_get_observations_respects_time_window(config: AppConfig, clock: FakeClock) -> None:
    api, _ = run_pipeline(config, clock)

    all_observations = api.get_observations(
        since=clock() - timedelta(seconds=config.window_seconds), until=clock(), now=clock()
    )
    narrow = api.get_observations(
        since=clock() - timedelta(seconds=0.6), until=clock(), now=clock()
    )

    assert all_observations
    assert 0 < len(narrow) <= len(all_observations)
    assert all(item.observed_at >= clock() - timedelta(seconds=0.6) for item in narrow)


def test_expired_observations_are_hidden_by_default(config: AppConfig, clock: FakeClock) -> None:
    api, _ = run_pipeline(config, clock)
    future = clock() + timedelta(seconds=config.observation_ttl_seconds + 1)

    assert api.get_observations(since=clock() - timedelta(seconds=10), now=future) == []
    assert (
        api.get_observations(
            since=clock() - timedelta(seconds=10), now=future, include_expired=True
        )
        != []
    )


def test_subscribe_events_receives_published_events(config: AppConfig, clock: FakeClock) -> None:
    pipeline = build_mock_pipeline(config, clock=clock)
    api = VisualContextApi.from_pipeline(pipeline)
    received: list[Event] = []
    api.subscribe_events(received.append)

    pipeline.run()

    assert [event.kind for event in received] == ["greeting-candidate"]


def test_turn_context_is_bounded_by_turn_window(config: AppConfig, clock: FakeClock) -> None:
    api, _ = run_pipeline(config, clock)
    turn_start = clock() - timedelta(seconds=1.2)

    turn = api.get_turn_context("turn-01", started_at=turn_start, ended_at=clock())

    assert turn.turn_id == "turn-01"
    assert turn.snapshot.window_start == turn_start
    assert turn.snapshot.window_end == clock()
    assert all(item.observed_at >= turn_start for item in turn.snapshot.observations)
    assert turn.snapshot.highlights
    scene = api.get_scene_snapshot(now=clock())
    assert scene.persons is not None
    assert turn.snapshot.persons == scene.persons
    assert scene.persons.current_person_count == 1
    assert scene.persons.recognizable_face_count is None

    outside = api.get_turn_context(
        "older",
        started_at=clock() - timedelta(seconds=20),
        ended_at=clock() - timedelta(seconds=10),
    )
    assert outside.snapshot.persons is None
