"""最小可运行示例：用 Mock 组件跑通 检测 -> 跟踪 -> 关系 -> 时间线 -> 快照 闭环。

运行：

    uv run python examples/run_mock_pipeline.py

示例演示三件事：
1. 通过工厂装配一条全 Mock 流水线（组件可替换）；
2. 订阅规则事件（迎宾候选）；
3. 输出有界、带有效期的场景快照 JSON，可直接作为 Agent 话轮上下文的形态参考。
"""

from __future__ import annotations

import json
from datetime import timedelta

from agent_visual_context.api import VisualContextApi
from agent_visual_context.config import load_config
from agent_visual_context.domain import BBox, Event
from agent_visual_context.logging_setup import configure_logging
from agent_visual_context.perception import MockTarget, RelationRule
from agent_visual_context.runtime import build_mock_pipeline


def main() -> None:
    configure_logging("WARNING")
    config = load_config(scene_id="scene-demo", max_frames=6, target_fps=2.0)

    pipeline = build_mock_pipeline(
        config,
        # 替换默认场景目标：一位顾客、一块大屏、一部手机
        targets=[
            MockTarget(
                label="person", bbox=BBox(x=80, y=120, width=120, height=260), confidence=0.93
            ),
            MockTarget(
                label="screen", bbox=BBox(x=320, y=60, width=280, height=200), confidence=0.9
            ),
            MockTarget(
                label="phone", bbox=BBox(x=150, y=240, width=40, height=70), confidence=0.71
            ),
        ],
        relation_rules=[
            RelationRule("person", "looking_at", "screen", confidence=0.84),
            RelationRule("person", "holding", "phone", confidence=0.66),
        ],
    )

    api = VisualContextApi.from_pipeline(pipeline)
    received: list[Event] = []
    api.subscribe_events(received.append)

    result = pipeline.run()
    snapshot = api.get_scene_snapshot(
        now=result.snapshot.generated_at if result.snapshot is not None else None
    )

    print("=== 运行状态 ===")
    print(f"状态：{pipeline.status.state.value} 帧数：{result.frames_processed}")
    for name, health in pipeline.status.components.items():
        print(f"  - {name:<9} {health.state.value:<8} {health.model_version}")

    print("\n=== 订阅到的规则事件 ===")
    for event in received:
        print(f"  {event.kind} 置信度={event.confidence:.2f} 有效期={event.expires_at.isoformat()}")

    print("\n=== 场景快照 JSON ===")
    print(json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2))

    print("\n=== 话轮上下文（最近 2 秒） ===")
    turn = api.get_turn_context(
        "turn-demo",
        started_at=snapshot.generated_at - timedelta(seconds=2),
        ended_at=snapshot.generated_at,
    )
    for highlight in turn.snapshot.highlights:
        print(f"  {highlight}")


if __name__ == "__main__":
    main()
