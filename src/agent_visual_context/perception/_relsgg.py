"""relsgg（RelateAnything 运行时）惰性加载。

真实关系推理依赖 `relsgg`（`relate-anything` extra），它进一步引入 torch、transformers 等重依赖，
且许可证为 AGPL-3.0（详见 `docs/research/relate-anything.md`）。为保持包导入轻量、
测试无需联网下载权重，只有实际构造真实后端时才加载 relsgg；缺失时抛出带
安装指引的 `PerceptionError`，由流水线降级处理，而不是让进程崩溃。
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ..errors import PerceptionError


def require_relsgg() -> ModuleType:
    """返回 relsgg 模块；未安装时抛出 `PerceptionError` 并给出安装指引。"""
    try:
        return importlib.import_module("relsgg")
    except ImportError as exc:
        msg = (
            "RelateAnything 关系推理依赖 relsgg，当前环境未安装。"
            "请执行 `uv sync --extra relate-anything`（会引入 torch/transformers 等重依赖，"
            "并需联网下载模型权重）后重试。许可证为 AGPL-3.0，商用前须完成合规确认。"
        )
        raise PerceptionError(msg) from exc
