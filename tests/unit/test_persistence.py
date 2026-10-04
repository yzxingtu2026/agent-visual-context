"""去抖与持续性单元测试：单帧抖动不得直接进入时间线。"""

from __future__ import annotations

from datetime import timedelta

from agent_visual_context.domain import Relation, TrackRef
from agent_visual_context.temporal import PersistenceGate
from tests.conftest import T0


def relation(*, offset_seconds: float, confidence: float = 0.8) -> Relation:
    observed_at = T0 + timedelta(seconds=offset_seconds)
    return Relation(
        subject=TrackRef(track_id="person-01", label="person"),
        predicate="looking_at",
        target=TrackRef(track_id="screen-01", label="screen"),
        confidence=confidence,
        observed_at=observed_at,
    )


def test_single_frame_relation_is_not_promoted() -> None:
    gate = PersistenceGate(min_hits=2, min_seconds=1.0)

    assert gate.update([relation(offset_seconds=0.0)], now=T0) == []


def test_sustained_relation_is_promoted_after_threshold() -> None:
    gate = PersistenceGate(min_hits=2, min_seconds=1.0)

    assert gate.update([relation(offset_seconds=0.0)], now=T0) == []
    stable = gate.update([relation(offset_seconds=1.0)], now=T0 + timedelta(seconds=1.0))

    assert len(stable) == 1
    assert stable[0].predicate == "looking_at"


def test_gap_longer_than_max_gap_resets_debounce() -> None:
    gate = PersistenceGate(min_hits=2, min_seconds=1.0, max_gap_seconds=1.5)

    gate.update([relation(offset_seconds=0.0)], now=T0)
    # 间隔 3 秒：超过 max_gap，之前的命中被丢弃，重新计数
    assert gate.update([relation(offset_seconds=3.0)], now=T0 + timedelta(seconds=3.0)) == []


def test_reset_clears_accumulated_hits() -> None:
    gate = PersistenceGate(min_hits=2, min_seconds=0.5)
    gate.update([relation(offset_seconds=0.0)], now=T0)
    gate.reset()

    assert gate.update([relation(offset_seconds=1.0)], now=T0 + timedelta(seconds=1.0)) == []


def test_latest_relation_is_promoted_with_recent_evidence() -> None:
    gate = PersistenceGate(min_hits=2, min_seconds=0.5)
    gate.update([relation(offset_seconds=0.0, confidence=0.5)], now=T0)

    stable = gate.update(
        [relation(offset_seconds=1.0, confidence=0.9)], now=T0 + timedelta(seconds=1.0)
    )

    assert stable[0].confidence == 0.9
    assert stable[0].observed_at == T0 + timedelta(seconds=1.0)
