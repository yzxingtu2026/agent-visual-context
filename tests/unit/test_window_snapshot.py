"""指定时间窗口快照服务单元测试：窗口边界、去重、过期过滤与场景回退。"""

from __future__ import annotations

from datetime import timedelta

from agent_visual_context.context import WindowSnapshotBuilder
from agent_visual_context.domain import Observation, TrackRef
from agent_visual_context.temporal import BoundedTimeline
from tests.conftest import T0


def make_observation(
    offset_seconds: float,
    *,
    subject: str = "person-01",
    ttl_seconds: float = 100.0,
    scene_id: str = "scene-window",
) -> Observation:
    observed_at = T0 + timedelta(seconds=offset_seconds)
    return Observation(
        scene_id=scene_id,
        subject=TrackRef(track_id=subject, label="person"),
        predicate="looking_at",
        target=TrackRef(track_id="screen-01", label="screen"),
        confidence=0.8,
        observed_at=observed_at,
        expires_at=observed_at + timedelta(seconds=ttl_seconds),
        source_id="mock-source",
    )


def timeline_with(*observations: Observation) -> BoundedTimeline:
    # default_ttl 设得足够大，避免覆盖各观察自带的 expires_at，让用例只验证窗口逻辑
    timeline = BoundedTimeline(capacity=32, default_ttl=timedelta(seconds=1000), clock=lambda: T0)
    for observation in observations:
        timeline.add_observation(observation)
    return timeline


def test_window_snapshot_bounds_and_filtering() -> None:
    # 每条使用不同 subject，避免去重折叠，专注验证窗口边界与过期过滤
    timeline = timeline_with(
        *(make_observation(float(i), subject=f"person-{i:02d}") for i in range(6))
    )
    builder = WindowSnapshotBuilder(clock=lambda: T0)

    snapshot = builder.build(
        timeline,
        start=T0 + timedelta(seconds=2),
        end=T0 + timedelta(seconds=4),
    )

    assert snapshot.window_start == T0 + timedelta(seconds=2)
    assert snapshot.window_end == T0 + timedelta(seconds=4)
    assert [item.observed_at for item in snapshot.observations] == [
        T0 + timedelta(seconds=2),
        T0 + timedelta(seconds=3),
        T0 + timedelta(seconds=4),
    ]


def test_window_snapshot_dedupes_same_relation() -> None:
    timeline = timeline_with(
        make_observation(0.0),
        make_observation(1.0),
        make_observation(2.0, subject="person-02"),
    )
    builder = WindowSnapshotBuilder(clock=lambda: T0)

    snapshot = builder.build(timeline, start=T0, end=T0 + timedelta(seconds=2))

    # person-01 的三条重复关系只保留最新一条；person-02 独立保留
    assert len(snapshot.observations) == 2
    subjects = {item.subject.track_id for item in snapshot.observations}
    assert subjects == {"person-01", "person-02"}


def test_window_snapshot_filters_expired() -> None:
    timeline = timeline_with(
        make_observation(0.0, ttl_seconds=1.0),
        make_observation(1.0, ttl_seconds=100.0),
    )
    builder = WindowSnapshotBuilder(clock=lambda: T0)

    snapshot = builder.build(timeline, start=T0, end=T0 + timedelta(seconds=5))

    # 第一条在窗口结束时刻已过期，被过滤
    assert [item.observed_at for item in snapshot.observations] == [T0 + timedelta(seconds=1)]


def test_window_snapshot_label_and_scene_fallback() -> None:
    timeline = timeline_with(make_observation(0.0))
    builder = WindowSnapshotBuilder(clock=lambda: T0)

    labeled = builder.build(timeline, start=T0, end=T0 + timedelta(seconds=1), label="turn-07")
    assert labeled.scene_id == "scene-window"
    assert "话轮 turn-07 期间" in labeled.highlights[0]

    empty = builder.build(
        timeline,
        start=T0 + timedelta(seconds=50),
        end=T0 + timedelta(seconds=60),
        scene_id="scene-explicit",
    )
    assert empty.scene_id == "scene-explicit"
    assert empty.observations == []
    assert "指定窗口内" in empty.highlights[0]
