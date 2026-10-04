"""ultralytics（YOLO-World 运行时）惰性加载。

真实目标检测依赖 `ultralytics`（`yolo-world` extra），它进一步引入 torch 等重依赖，
且许可证为 AGPL-3.0（详见 `docs/research/yolo-world.md`）。为保持包导入轻量、
测试无需联网下载权重，只有实际构造真实后端时才加载 ultralytics；缺失时抛出带
安装指引的 `PerceptionError`，由流水线降级处理，而不是让进程崩溃。
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ..errors import PerceptionError


def require_ultralytics() -> ModuleType:
    """返回 ultralytics 模块；未安装时抛出 `PerceptionError` 并给出安装指引。"""
    try:
        return importlib.import_module("ultralytics")
    except ImportError as exc:
        msg = (
            "YOLO-World 检测依赖 ultralytics，当前环境未安装。"
            "请执行 `uv sync --extra yolo-world`（会引入 torch 等重依赖，"
            "并需联网下载模型权重）后重试。许可证为 AGPL-3.0，商用前须完成合规确认。"
        )
        raise PerceptionError(msg) from exc
