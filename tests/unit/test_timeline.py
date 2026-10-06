"""有界时间线单元测试：容量约束、TTL 裁剪与窗口查询。"""

from __future__ import annotations

from datetime import timedelta

from agent_visual_context.domain import (
    Event,
    Observation,
    PersonCountQuality,
    PersonSceneSummary,
    TrackRef,
)
from agent_visual_context.temporal import BoundedTimeline
from tests.conftest import T0


def make_observation(offset_seconds: float, *, ttl_seconds: float = 4.0) -> Observation:
    observed_at = T0 + timedelta(seconds=offset_seconds)
    return Observation(
        scene_id="scene-test",
        subject=TrackRef(track_id=f"person-{offset_seconds:.0f}", label="person"),
        predicate="looking_at",
        target=TrackRef(track_id="screen-01", label="screen"),
        confidence=0.8,
        observed_at=observed_at,
        expires_at=observed_at + timedelta(seconds=ttl_seconds),
        source_id="mock-source",
    )


def make_event(offset_seconds: float) -> Event:
    observed_at = T0 + timedelta(seconds=offset_seconds)
    return Event(
        scene_id="scene-test",
        kind="greeting-candidate",
        observed_at=observed_at,
        expires_at=observed_at + timedelta(seconds=30),
    )


def test_timeline_is_bounded_by_capacity() -> None:
    timeline = BoundedTimeline(capacity=3, clock=lambda: T0)

    for index in range(6):
        timeline.add_observation(make_observation(float(index)))

    assert len(timeline.observations()) == 3
    assert timeline.dropped_total == 3


def test_add_observation_recomputes_expiry_from_ttl() -> None:
    timeline = BoundedTimeline(capacity=8, default_ttl=timedelta(seconds=2), clock=lambda: T0)

    stored = timeline.add_observation(make_observation(0.0, ttl_seconds=99.0))

    assert stored.expires_at == T0 + timedelta(seconds=2)


def test_add_observation_keeps_shorter_expiry() -> None:
    timeline = BoundedTimeline(capacity=8, default_ttl=timedelta(seconds=10), clock=lambda: T0)

    stored = timeline.add_observation(make_observation(0.0, ttl_seconds=1.0))

    assert stored.expires_at == T0 + timedelta(seconds=1)


def test_prune_removes_expired_items_only() -> None:
    timeline = BoundedTimeline(capacity=8, clock=lambda: T0)
    timeline.add_observation(make_observation(0.0, ttl_seconds=1.0))
    timeline.add_observation(make_observation(1.0, ttl_seconds=10.0))
    timeline.add_event(make_event(0.0))

    removed = timeline.prune(now=T0 + timedelta(seconds=3))

    assert removed == 1
    assert len(timeline.observations()) == 1
    assert len(timeline.events()) == 1


def test_window_query_filters_by_observed_at_and_expiry() -> None:
    timeline = BoundedTimeline(capacity=16, clock=lambda: T0)
    for index in range(6):
        timeline.add_observation(make_observation(float(index), ttl_seconds=100.0))

    window = timeline.observations(
        start=T0 + timedelta(seconds=2),
        end=T0 + timedelta(seconds=4),
        now=T0 + timedelta(seconds=4),
    )

    assert [item.observed_at for item in window] == [
        T0 + timedelta(seconds=2),
        T0 + timedelta(seconds=3),
        T0 + timedelta(seconds=4),
    ]


def test_person_summary_capacity_ttl_and_latest_window() -> None:
    timeline = BoundedTimeline(capacity=2, default_ttl=timedelta(seconds=2))
    for offset in range(3):
        sampled_at = T0 + timedelta(seconds=offset)
        timeline.add_person_summary(
            PersonSceneSummary(
                scene_id="scene-test",
                source_id="camera",
                sampled_at=sampled_at,
                expires_at=sampled_at + timedelta(seconds=99),
                current_person_count=offset,
                quality=PersonCountQuality.DETECTED,
            )
        )

    assert timeline.dropped_total == 1
    latest = timeline.latest_person_summary(
        start=T0 + timedelta(seconds=1),
        end=T0 + timedelta(seconds=2),
        now=T0 + timedelta(seconds=2),
    )
    assert latest is not None and latest.current_person_count == 2
    assert latest.expires_at == T0 + timedelta(seconds=4)
    assert (
        timeline.latest_person_summary(
            start=T0, end=T0 + timedelta(seconds=2), now=T0 + timedelta(seconds=4)
        )
        is None
    )
