"""Experiment base class + Registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar


@dataclass
class ExperimentResult:
    name: str
    category: str
    params: dict[str, Any]
    metrics: dict[str, float]
    started_at: str
    finished_at: str
    duration_s: float
    status: str = "done"
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)


class Experiment(ABC):
    name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    def run(self, **kwargs) -> ExperimentResult:
        ...

    def default_params(self) -> dict[str, Any]:
        return {}


class Registry:
    _experiments: dict[str, type[Experiment]] = {}

    @classmethod
    def register(cls, exp_cls: type[Experiment]) -> type[Experiment]:
        name = exp_cls.name or exp_cls.__name__
        cls._experiments[name] = exp_cls
        return exp_cls

    @classmethod
    def get(cls, name: str) -> type[Experiment] | None:
        return cls._experiments.get(name)

    @classmethod
    def list(cls, category: str | None = None) -> list[str]:
        if category:
            return [n for n, c in cls._experiments.items() if c.category == category]
        return list(cls._experiments.keys())

    @classmethod
    def categories(cls) -> set[str]:
        return {c.category for c in cls._experiments.values()}

    @classmethod
    def run_all(cls, **kwargs) -> list[ExperimentResult]:
        results = []
        for name, exp_cls in cls._experiments.items():
            exp = exp_cls()
            results.append(exp.run(**kwargs))
        return results
