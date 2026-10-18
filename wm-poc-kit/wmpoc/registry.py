"""String-keyed registries. Components are selected by the `type:` key in the config."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, type] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if key in self._items:
                raise KeyError(f"{self.name} {key!r} registered twice")
            self._items[key] = cls
            cls.registry_key = key  # type: ignore[attr-defined]
            return cls

        return deco

    def get(self, key: str) -> type:
        _import_components()
        if key not in self._items:
            raise KeyError(f"unknown {self.name} {key!r}; available: {sorted(self._items)}")
        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(key)(*args, **kwargs)

    def keys(self) -> list[str]:
        _import_components()
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        _import_components()
        return key in self._items


def _import_components() -> None:
    # Importing the packages runs the @register decorators of every module inside them.
    import wmpoc.models  # noqa: F401


ENCODERS, WORLD_MODELS, HEADS = Registry("encoder"), Registry("world_model"), Registry("head")
