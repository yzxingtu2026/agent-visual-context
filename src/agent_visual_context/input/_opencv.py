"""OpenCV 惰性加载。

真实图片/视频解码依赖 `opencv-python`（`vision` extra），但包导入不强制要求它：
只有实际打开图片/视频输入源时才加载 cv2，缺失时抛出带安装指引的 `FrameSourceError`。
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ..errors import FrameSourceError


def require_cv2() -> ModuleType:
    """返回 cv2 模块；未安装时抛出 `FrameSourceError` 并给出安装指引。"""
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        msg = (
            "图片/视频输入依赖 OpenCV，当前环境未安装。"
            "请执行 `uv sync --extra vision`（或 pip install opencv-python）后重试。"
        )
        raise FrameSourceError(msg) from exc
