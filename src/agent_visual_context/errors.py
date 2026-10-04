"""统一错误类型。

约定：
- 可恢复的组件异常（检测器超时、推理器报错等）由 `runtime` 捕获并转为降级状态，不向上抛出；
- 不可恢复的配置/编程错误直接抛出，由 CLI 统一转换为退出码。
"""

from __future__ import annotations


class AgentVisualContextError(Exception):
    """本项目所有异常的基类。"""


class ConfigurationError(AgentVisualContextError):
    """配置缺失或非法。"""


class FrameSourceError(AgentVisualContextError):
    """输入源打开、读取或释放失败。"""


class PerceptionError(AgentVisualContextError):
    """检测、跟踪或关系推理失败，通常可降级继续运行。"""


class PipelineError(AgentVisualContextError):
    """流水线编排错误。"""
