"""离线素材示例：用图片和视频跑通 Mock 视觉处理闭环。

运行（默认使用 tests/fixtures 下的固定素材）：

    uv run python examples/run_offline_pipeline.py
    uv run python examples/run_offline_pipeline.py path/to/image.png
    uv run python examples/run_offline_pipeline.py path/to/video.mp4

示例演示三件事：
1. 图片与视频入口复用同一套 FrameSource -> 流水线 -> 时间线 -> 快照逻辑；
2. 视频按 target_fps 采样，帧时间戳统一映射到时间轴；
3. 输出结构化 JSON 快照，而不是日志文本。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_visual_context.api import VisualContextApi
from agent_visual_context.config import load_config
from agent_visual_context.logging_setup import configure_logging
from agent_visual_context.runtime import build_offline_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATERIALS = [
    REPO_ROOT / "tests" / "fixtures" / "sample_image.png",
    REPO_ROOT / "tests" / "fixtures" / "sample_video.mp4",
]


def run_one(path: Path) -> None:
    configure_logging("WARNING")
    # 单帧图片放宽去抖门槛，保证一张图也能产出观察
    config = load_config(
        scene_id=f"scene-{path.stem}",
        source_id=path.stem,
        target_fps=2.0,
        max_frames=0,  # 0 表示不限制，处理到素材耗尽
        persistence_min_hits=1,
        persistence_min_seconds=0.0,
    )
    pipeline = build_offline_pipeline(config, path)
    api = VisualContextApi.from_pipeline(pipeline)

    result = pipeline.run()
    snapshot = api.get_scene_snapshot(
        now=result.snapshot.generated_at if result.snapshot is not None else None
    )

    print(f"\n=== {path.name} ===")
    print(f"状态：{pipeline.status.state.value} 处理帧数：{result.frames_processed}")
    print(f"观察：{len(result.observations)} 事件：{len(result.events)}")
    for event in result.events:
        print(f"  事件 {event.kind} payload={json.dumps(event.payload, ensure_ascii=False)}")
    print("快照 JSON：")
    print(json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    materials = [Path(arg) for arg in sys.argv[1:]] or DEFAULT_MATERIALS
    for path in materials:
        run_one(path)


if __name__ == "__main__":
    main()
