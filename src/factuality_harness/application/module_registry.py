"""Lifecycle-aware registry of domain modules.

Only ACTIVE modules are allowed to influence final answers. SHADOW modules run
in parallel for evaluation only and DRAFT modules are disabled by default.
This registry is the single source of truth for the application layer.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..domain.modules import ModuleSpec, ModuleStatus


class ModuleRegistry:
    def __init__(self, modules: Iterable["BaseDomainModule"] | None = None) -> None:  # noqa: F821
        self._modules: dict[str, "BaseDomainModule"] = {}  # noqa: F821
        for m in modules or []:
            self.register(m)

    def register(self, module: "BaseDomainModule") -> None:  # noqa: F821
        self._modules[module.spec.name] = module

    def get(self, name: str) -> "BaseDomainModule | None":  # noqa: F821
        return self._modules.get(name)

    def all(self) -> list["BaseDomainModule"]:  # noqa: F821
        return list(self._modules.values())

    def active(self) -> list["BaseDomainModule"]:  # noqa: F821
        return [m for m in self._modules.values() if m.spec.status == ModuleStatus.ACTIVE]

    def shadow(self) -> list["BaseDomainModule"]:  # noqa: F821
        return [m for m in self._modules.values() if m.spec.status == ModuleStatus.SHADOW]

    def specs(self) -> list[ModuleSpec]:
        return [m.spec for m in self._modules.values()]

    def promote(self, name: str, status: ModuleStatus) -> None:
        m = self._modules.get(name)
        if m is None:
            raise KeyError(f"No module named {name!r}.")
        m.spec = m.spec.model_copy(update={"status": status})
