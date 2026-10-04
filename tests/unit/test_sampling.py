"""单元测试：帧采样策略。"""

from __future__ import annotations

import pytest

from agent_visual_context.input import plan_sampling


def test_empty_total_yields_empty_plan() -> None:
    assert plan_sampling(0, source_fps=10.0, target_fps=2.0) == []
    assert plan_sampling(-3, source_fps=10.0, target_fps=2.0) == []


def test_target_not_below_source_keeps_all_frames() -> None:
    assert plan_sampling(5, source_fps=2.0, target_fps=10.0) == [0, 1, 2, 3, 4]
    assert plan_sampling(5, source_fps=10.0, target_fps=10.0) == [0, 1, 2, 3, 4]


def test_unknown_source_fps_keeps_all_frames() -> None:
    assert plan_sampling(4, source_fps=0.0, target_fps=2.0) == [0, 1, 2, 3]


def test_downsampling_is_uniform_and_starts_at_first_frame() -> None:
    # 10fps -> 2fps：步长 5，20 帧取 4 帧
    assert plan_sampling(20, source_fps=10.0, target_fps=2.0) == [0, 5, 10, 15]
    # 30fps -> 10fps：步长 3
    assert plan_sampling(10, source_fps=30.0, target_fps=10.0) == [0, 3, 6, 9]


def test_non_integer_step_has_no_duplicates_and_stays_in_range() -> None:
    plan = plan_sampling(17, source_fps=24.0, target_fps=5.0)

    assert plan[0] == 0
    assert all(0 <= index < 17 for index in plan)
    assert len(plan) == len(set(plan))
    assert plan == sorted(plan)


def test_invalid_target_fps_raises() -> None:
    with pytest.raises(ValueError, match="target_fps"):
        plan_sampling(10, source_fps=10.0, target_fps=0.0)
