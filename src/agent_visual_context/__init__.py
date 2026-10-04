"""面向 Agent 的实时视觉关系上下文处理器。

包内依赖方向（详见 docs/architecture/module-boundaries.md）：

domain <- input/perception/temporal/context/policies <- runtime <- api/cli

`domain` 是最内层，只依赖标准库与 Pydantic，不得引用 OpenCV、具体模型或 Agent 框架。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
