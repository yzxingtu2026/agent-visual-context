"""策略层边界：规则派生事件（如迎宾候选）。"""

from __future__ import annotations

from .base import Policy
from .greeting import GreetingCandidatePolicy

__all__ = ["GreetingCandidatePolicy", "Policy"]
