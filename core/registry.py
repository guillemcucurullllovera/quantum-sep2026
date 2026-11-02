"""Generic plugin registry + three concrete registries (algorithms, backends,
applications). Modules register themselves via decorator at import time;
core.cli and experiments/runners discover what's available through these
rather than hardcoded import lists.
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

from core.errors import ValidationError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind_name: str) -> None:
        self._kind_name = kind_name
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValidationError(
                    f"{self._kind_name} '{name}' is already registered "
                    f"(registering twice usually means a copy-paste of an "
                    f"existing module without renaming it)."
                )
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as e:
            raise ValidationError(
                f"Unknown {self._kind_name} '{name}'. Registered: {sorted(self._items)}"
            ) from e

    def list(self) -> list[str]:
        return sorted(self._items)


ALGORITHM_REGISTRY: Registry = Registry("algorithm")
BACKEND_REGISTRY: Registry = Registry("backend")
APPLICATION_REGISTRY: Registry = Registry("application")
