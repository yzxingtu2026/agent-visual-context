"""本地摄像头实时链路示例：持续采集 -> 有界保最新帧 -> 低频推理 -> 健康/指标/快照。

运行（需要本机摄像头，并安装视觉依赖 `uv sync --extra vision`）：

    uv run python examples/run_camera_live.py                       # 默认设备 0，运行 5s
    uv run python examples/run_camera_live.py --device 1 --duration 8 --fps 2

示例演示三件事：
1. 采集/消费解耦：后台线程持续采集，主循环按 target_fps 处理最新帧，缓冲不会无限堆积；
2. 视觉降级不阻塞宿主：无摄像头或设备被占用时返回 degraded 状态，而不是抛异常中断；
3. 输出健康状态、运行指标与结构化场景快照，供上层 Agent/宿主链路读取。

所有结果均为视觉辅助观察，不触发任何业务写操作。
"""

from __future__ import annotations

import argparse
import json

from agent_visual_context.api import VisualContextApi
from agent_visual_context.config import load_config
from agent_visual_context.logging_setup import configure_logging
from agent_visual_context.runtime import build_live_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="本地摄像头实时视觉链路示例")
    parser.add_argument("--device", type=int, default=0, help="摄像头设备索引，默认 0")
    parser.add_argument("--duration", type=float, default=5.0, help="运行时长（秒），默认 5")
    parser.add_argument("--fps", type=float, default=2.0, help="采样帧率，默认 2.0")
    parser.add_argument("--width", type=int, default=640, help="请求分辨率宽，默认 640")
    parser.add_argument("--height", type=int, default=480, help="请求分辨率高，默认 480")
    args = parser.parse_args()

    configure_logging("INFO")
    config = load_config(
        scene_id="scene-camera-demo",
        target_fps=args.fps,
        camera_device_index=args.device,
        camera_width=args.width,
        camera_height=args.height,
        live_duration_seconds=args.duration,
        # 放宽去抖门槛，便于短演示也能产出观察
        persistence_min_hits=1,
        persistence_min_seconds=0.0,
    )

    runtime = build_live_runtime(config)
    api = VisualContextApi.from_live_runtime(runtime)
    result = runtime.run(duration_seconds=args.duration)

    health = api.get_health()
    metrics = api.get_metrics()

    print(f"\n=== 摄像头实时链路（device={args.device}）===")
    print(f"状态：{result.status.state.value} 采集降级：{'是' if runtime.capture_failed else '否'}")
    if health is not None:
        print("组件状态：")
        for name, comp in health.components.items():
            detail = comp.model_version
            if comp.last_error:
                detail += f" 最近错误={comp.last_error}"
            print(f"  - {name:<9} {comp.state.value:<8} {detail}")
    if metrics is not None:
        print(
            f"指标：运行 {metrics.uptime_seconds:.1f}s 采集 {metrics.frames_captured} "
            f"处理 {metrics.frames_processed} 丢弃 {metrics.frames_dropped} "
            f"有效FPS {metrics.effective_fps:.2f} 丢帧率 {metrics.drop_rate:.2f} "
            f"失败率 {metrics.failure_rate:.2f} 平均延迟 {metrics.mean_latency_seconds:.3f}s"
        )
    if result.snapshot is not None:
        print("快照 JSON：")
        print(json.dumps(result.snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print("提示：以上均为视觉辅助观察，不触发任何业务写操作。")


if __name__ == "__main__":
    main()
