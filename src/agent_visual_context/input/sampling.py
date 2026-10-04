"""帧采样策略。

离线素材（图片序列、视频文件）按目标帧率统一采样，
采样结果只包含被选中的原始帧编号，时间戳换算由具体输入源负责。
"""

from __future__ import annotations


def plan_sampling(total_frames: int, *, source_fps: float, target_fps: float) -> list[int]:
    """返回按 `target_fps` 从 `total_frames` 帧中采样的原始帧编号列表。

    规则：
    - `total_frames <= 0` 返回空计划；
    - `source_fps <= 0`（未知帧率）或目标帧率不低于源帧率时，逐帧全选；
    - 否则按 `source_fps / target_fps` 步长等间隔取整，保证首帧必选且编号不重复。
    """
    if total_frames <= 0:
        return []
    if target_fps <= 0:
        msg = f"target_fps 必须为正数，收到 {target_fps}"
        raise ValueError(msg)
    if source_fps <= 0 or source_fps <= target_fps:
        return list(range(total_frames))

    step = source_fps / target_fps
    selected: list[int] = []
    seen: set[int] = set()
    position = 0.0
    while True:
        index = int(round(position))
        if index >= total_frames:
            break
        if index not in seen:
            seen.add(index)
            selected.append(index)
        position += step
    return selected
