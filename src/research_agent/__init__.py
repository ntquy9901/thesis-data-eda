"""Research Agent — tự động nghiên cứu và thử nghiệm phương pháp phân tích tin tức tài chính.

Continuous loop:
  research() → experiment() → log() → sleep() → research()

Mỗi experiment là một method được đăng ký trong registry, chạy trên dataset
và ghi kết quả vào SQLite DB để so sánh cross-method.
"""

from src.research_agent.base import Experiment, Registry
from src.research_agent.runner import ResearchAgent

__all__ = ["Experiment", "Registry", "ResearchAgent"]
