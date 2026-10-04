"""运行时日志配置。

规范：运行时日志一律使用标准 `logging`，禁止用零散 `print` 代替 logger。
面向终端用户的命令输出（CLI 报告）不在此约束内。
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

LOGGER_NAME = "agent_visual_context"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """返回项目命名空间下的 logger。"""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> None:
    """配置项目根 logger。

    幂等：重复调用只更新级别与格式，不会叠加 handler。
    """
    logger = logging.getLogger(LOGGER_NAME)
    resolved_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logger.setLevel(resolved_level)

    formatter = logging.Formatter(DEFAULT_FORMAT)
    if not logger.handlers:
        handler: logging.Handler = logging.StreamHandler(
            stream if stream is not None else sys.stderr
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(formatter)

    logger.propagate = False
