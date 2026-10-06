"""Model catalog: a data-driven registry of routable models.

Ships a default catalog (see data/default_catalog.json) and merges user files on
top. The same open-weight model may appear under several providers; policies pick
the cheapest eligible (provider, price) pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .types import ConfigError, ModelSpec, Tier

DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "default_catalog.json"


def _spec_from_dict(d: dict[str, Any]) -> ModelSpec:
    required = ("id", "provider")
    missing = [k for k in required if not d.get(k)]
    if missing:
        raise ConfigError(f"catalog entry missing required fields {missing}: {d}")
    tier = d.get("tier", "balanced")
    try:
        tier_enum = Tier(tier)
    except ValueError as e:
        raise ConfigError(f"unknown tier {tier!r} in catalog entry {d['id']}") from e
    return ModelSpec(
        id=d["id"],
        provider=d["provider"],
        api=d.get("api", "openai-chat"),
        kind=d.get("kind", "chat"),
        tier=tier_enum,
        base_url=d.get("base_url"),
        api_key_env=d.get("api_key_env"),
        input_price=float(d.get("input_price", 0.0)),
        output_price=float(d.get("output_price", 0.0)),
        context_window=int(d.get("context_window", 128_000)),
        max_output_tokens=int(d.get("max_output_tokens", 8_192)),
        capabilities=frozenset(d.get("capabilities", [])),
        intelligence=float(d.get("intelligence", 0.0)),
        aliases=tuple(d.get("aliases", [])),
        notes=d.get("notes", ""),
    )


class ModelCatalog:
    def __init__(self, specs: Iterable[ModelSpec] | None = None) -> None:
        self._specs: dict[tuple[str, str], ModelSpec] = {}
        for spec in specs or []:
            self.register(spec)

    # -- registration ----------------------------------------------------

    def register(self, spec: ModelSpec, replace: bool = True) -> None:
        if spec.key in self._specs and not replace:
            raise ConfigError(f"model {spec.label()} already registered")
        self._specs[spec.key] = spec

    def add_dict(self, data: dict[str, Any]) -> None:
        for entry in data.get("models", []):
            self.register(_spec_from_dict(entry))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCatalog":
        catalog = cls()
        catalog.add_dict(data)
        return catalog

    @classmethod
    def load(cls, path: str | Path | None = None, merge_defaults: bool = True) -> "ModelCatalog":
        catalog = cls()
        if merge_defaults:
            catalog.add_dict(json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8")))
        if path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            catalog.add_dict(data)  # user entries override defaults
        return catalog

    # -- queries ----------------------------------------------------------

    def all(self, kind: str | None = None) -> list[ModelSpec]:
        return [s for s in self._specs.values() if kind is None or s.kind == kind]

    def get(self, id: str, provider: str | None = None) -> ModelSpec | None:
        if provider:
            return self._specs.get((provider, id))
        matches = [s for s in self._specs.values() if s.id == id or id in s.aliases]
        return matches[0] if matches else None

    def require(self, id: str, provider: str | None = None) -> ModelSpec:
        spec = self.get(id, provider)
        if spec is None:
            raise ConfigError(f"model {id!r} (provider={provider!r}) not in catalog")
        return spec

    def by_tier(self, tier: Tier, kind: str = "chat") -> list[ModelSpec]:
        """Models in a tier, cheapest (blended 1k-in/256-out) first."""
        specs = [s for s in self.all(kind=kind) if s.tier == tier]
        return sorted(specs, key=lambda s: s.expected_cost(1_000, 256))

    def capable(
        self,
        required_capabilities: Iterable[str] = (),
        min_ctx: int = 0,
        min_intelligence: float = 0.0,
        kind: str = "chat",
    ) -> list[ModelSpec]:
        """Eligible models sorted by expected cost (cheapest first)."""
        caps = frozenset(required_capabilities)
        eligible = [
            s
            for s in self.all(kind=kind)
            if caps <= s.capabilities
            and s.context_window >= min_ctx
            and s.intelligence >= min_intelligence
        ]
        return sorted(eligible, key=lambda s: s.expected_cost(1_000, 256))

    def strongest(self, kind: str = "chat") -> ModelSpec:
        specs = self.all(kind=kind)
        if not specs:
            raise ConfigError("catalog has no models")
        return max(specs, key=lambda s: (s.intelligence, -s.expected_cost(1_000, 256)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "id": s.id,
                    "provider": s.provider,
                    "api": s.api,
                    "kind": s.kind,
                    "tier": s.tier.value,
                    "base_url": s.base_url,
                    "api_key_env": s.api_key_env,
                    "input_price": s.input_price,
                    "output_price": s.output_price,
                    "context_window": s.context_window,
                    "max_output_tokens": s.max_output_tokens,
                    "capabilities": sorted(s.capabilities),
                    "intelligence": s.intelligence,
                    "aliases": list(s.aliases),
                    "notes": s.notes,
                }
                for s in self.all()
            ]
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
