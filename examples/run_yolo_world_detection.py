"""YOLO-World 离线检测示例：在固定图片/视频上验证检测框、类别、置信度与耗时。

运行前需安装重依赖并联网下载权重（详见 docs/research/yolo-world.md）：

    uv sync --extra yolo-world
    uv run python examples/run_yolo_world_detection.py                 # 默认 tests/fixtures 素材
    uv run python examples/run_yolo_world_detection.py path/to/img.png
    uv run python examples/run_yolo_world_detection.py path/to/video.mp4

示例聚焦"检测"这一层：直接驱动 `YoloWorldDetector` 逐帧输出统一 Detection 与可观测指标，
演示模型不可用（未安装 ultralytics / 权重缺失）时如何降级为提示而非崩溃。
关系推理与时间线闭环见 examples/run_offline_pipeline.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_visual_context.config import load_config
from agent_visual_context.errors import PerceptionError
from agent_visual_context.input import frame_source_from_path
from agent_visual_context.logging_setup import configure_logging, get_logger
from agent_visual_context.perception import YoloWorldDetector

logger = get_logger("examples.yolo_world")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATERIALS = [
    REPO_ROOT / "tests" / "fixtures" / "sample_image.png",
    REPO_ROOT / "tests" / "fixtures" / "sample_video.mp4",
]


def detect_material(path: Path, detector: YoloWorldDetector) -> None:
    print(f"\n=== {path.name} ===")
    source = frame_source_from_path(path, target_fps=2.0)
    source.open()
    try:
        while (frame := source.read()) is not None:
            try:
                detections = detector.detect(frame)
            except PerceptionError as exc:
                # 模型不可用/推理失败：降级为提示，不中断示例
                print(f"  [降级] {frame.frame_id}: {exc}")
                return
            print(
                f"  帧 {frame.frame_id}（{frame.width}x{frame.height}）检出 {len(detections)} 个目标"
            )
            for item in detections:
                box = item.bbox
                print(
                    f"    - {item.label:<8} conf={item.confidence:.2f} "
                    f"bbox=({box.x:.0f},{box.y:.0f},{box.width:.0f},{box.height:.0f})"
                )
    finally:
        source.close()

    stats = detector.stats
    print(
        f"  统计：帧={stats.frames} 检测={stats.detections} "
        f"延迟(min/mean/max)={stats.min_latency_seconds or 0:.3f}/"
        f"{stats.mean_latency_seconds:.3f}/{stats.max_latency_seconds or 0:.3f}s"
    )


def main() -> None:
    configure_logging("WARNING")
    materials = [Path(arg) for arg in sys.argv[1:]] or DEFAULT_MATERIALS
    config = load_config(
        detector_backend="yolo-world",
        detector_classes=["person", "hand", "phone", "screen"],
        detector_device="cpu",
    )
    detector = YoloWorldDetector.from_config(config)
    print(f"模型版本：{detector.model_version}  类别：{list(detector.settings.classes)}")
    for path in materials:
        detect_material(path, detector)
    detector.close()


if __name__ == "__main__":
    main()
