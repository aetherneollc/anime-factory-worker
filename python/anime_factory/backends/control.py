"""Control backends: single-ref identity (Phase B). Pluggable — not Kolors-only."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from anime_factory.contracts import CharacterRefError, assert_character_refs, is_master_ref

CONTROL_BACKENDS = ("kolors_ipadapter", "ip_adapter", "composite", "none")
DEFAULT_CONTROL_BACKEND = "kolors_ipadapter"


class ControlBackendError(ValueError):
    """CONTROL_BACKEND invalid or request violates single-ref master contract."""


@dataclass
class ControlRequest:
    prompt: str
    negative_prompt: str = ""
    character_refs: list[str] = field(default_factory=list)
    control_mode: str = "identity"
    reference_pngs: Sequence[bytes] = ()
    width: int = 1344
    height: int = 768
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.character_refs = assert_character_refs(self.character_refs)
        if len(self.character_refs) > 1:
            raise ControlBackendError(
                "Phase B control accepts a single master.png ref; "
                f"got {len(self.character_refs)} refs"
            )
        for ref in self.character_refs:
            if not is_master_ref(ref):
                raise CharacterRefError(f"control ref must be master.png: {ref!r}")


@dataclass
class ControlResult:
    png: bytes
    backend: str
    control_mode: str
    used_refs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class ControlBackend(ABC):
    name: str = "base"

    @abstractmethod
    def apply(self, request: ControlRequest) -> ControlResult:
        raise NotImplementedError


def select_control_backend(env: Mapping[str, str] | None = None) -> str:
    mapping = env if env is not None else os.environ
    raw = str(mapping.get("CONTROL_BACKEND") or DEFAULT_CONTROL_BACKEND).strip().lower()
    if raw not in CONTROL_BACKENDS:
        raise ControlBackendError(
            f"CONTROL_BACKEND must be one of {CONTROL_BACKENDS}, got {raw!r}"
        )
    return raw


class NoneControlBackend(ControlBackend):
    """No-op: Control disabled (tier A)."""

    name = "none"

    def apply(self, request: ControlRequest) -> ControlResult:
        raise ControlBackendError("CONTROL_BACKEND=none cannot apply identity control")


class KolorsIpAdapterControlBackend(ControlBackend):
    """Default Phase-B identity Control stub (Kolors IP-Adapter shape).

    Real weights stay on the Comfy Kolors path; this stub validates the
    single-master contract and returns a marker PNG for unit tests / dry runs.
    """

    name = "kolors_ipadapter"

    def apply(self, request: ControlRequest) -> ControlResult:
        if request.control_mode not in {"identity", "pose", "depth", "compose"}:
            raise ControlBackendError(f"unsupported control_mode: {request.control_mode!r}")
        if request.control_mode != "identity":
            raise ControlBackendError(
                "Phase B stub only implements identity control; "
                f"got {request.control_mode!r}"
            )
        if not request.character_refs and not request.reference_pngs:
            raise ControlBackendError("identity control requires one master.png ref or PNG bytes")
        # Dry stub PNG (valid header only) — production wires Comfy IP-Adapter.
        png = _stub_png(request.width, request.height, tag=b"kolors-ipadapter-stub")
        return ControlResult(
            png=png,
            backend=self.name,
            control_mode="identity",
            used_refs=list(request.character_refs),
            meta={"stub": True, **dict(request.meta)},
        )


class IpAdapterControlBackend(ControlBackend):
    """Generic IP-Adapter slot — same single-ref contract, different adapter name."""

    name = "ip_adapter"

    def apply(self, request: ControlRequest) -> ControlResult:
        # Delegate contract checks to the Kolors stub shape for now.
        inner = KolorsIpAdapterControlBackend().apply(request)
        return ControlResult(
            png=inner.png,
            backend=self.name,
            control_mode=inner.control_mode,
            used_refs=inner.used_refs,
            meta={**inner.meta, "adapter": "ip_adapter"},
        )


class CompositeControlBackend(ControlBackend):
    """Phase-C style compose / lock stub — spatial first, not Control→2.1 refine."""

    name = "composite"

    def apply(self, request: ControlRequest) -> ControlResult:
        if not request.character_refs and not request.reference_pngs:
            raise ControlBackendError("composite control requires master.png ref or PNG bytes")
        png = _stub_png(request.width, request.height, tag=b"composite-control-stub")
        return ControlResult(
            png=png,
            backend=self.name,
            control_mode=request.control_mode or "compose",
            used_refs=list(request.character_refs),
            meta={"stub": True, "path": "compose_lock_inpaint", **dict(request.meta)},
        )


def _stub_png(width: int, height: int, *, tag: bytes) -> bytes:
    """Minimal PNG-like payload for stubs (tests use contracts, not decode)."""
    # Keep under MIN_STILL_BYTES intentionally — stubs are not shippable stills.
    header = b"\x89PNG\r\n\x1a\n"
    body = tag + f":{width}x{height}".encode("ascii")
    return header + body + b"\x00" * max(0, 64 - len(header) - len(body))


_BACKENDS: dict[str, type[ControlBackend]] = {
    "kolors_ipadapter": KolorsIpAdapterControlBackend,
    "ip_adapter": IpAdapterControlBackend,
    "composite": CompositeControlBackend,
    "none": NoneControlBackend,
}


def get_control_backend(name: str | None = None, *, env: Mapping[str, str] | None = None) -> ControlBackend:
    chosen = (name or select_control_backend(env=env)).strip().lower()
    if chosen not in _BACKENDS:
        raise ControlBackendError(
            f"CONTROL_BACKEND must be one of {CONTROL_BACKENDS}, got {chosen!r}"
        )
    return _BACKENDS[chosen]()
