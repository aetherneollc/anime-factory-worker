"""Stable Asset / Shot / Keyframe / Video / QC contracts for the stills pipeline.

Models are pluggable; these shapes must not drift with IMAGE_BACKEND /
CONTROL_BACKEND / VIDEO_BACKEND swaps.

B-tier hard rule: ``character_refs`` are production Master paths only
(``…/master.png``). Sheets stay in ``sheet_refs`` (docs / registry) and must
not be fed into Control or Video by default. Workers must not scan a whole
character directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal, Mapping, Sequence

MASTER_FILENAME = "master.png"
KEYFRAME_FINAL_NAME = "keyframe_final.png"

ShotTier = Literal["A", "B", "C"]
ControlMode = Literal["identity", "pose", "depth", "compose", "none"]
QcVerdict = Literal["pass", "fail", "skip"]
KeyframeStatus = Literal["pending", "generated", "qc_pass", "qc_fail"]
VideoStatus = Literal["blocked", "pending", "generating", "ready", "failed"]


class CharacterRefError(ValueError):
    """character_refs must be master.png paths only — never sheets or dirs."""


class KeyframeQcGateError(RuntimeError):
    """Video must not run when Keyframe QC failed or is missing."""


def _posix(path: str | Path) -> str:
    text = str(path or "").replace("\\", "/").strip()
    return text.rstrip("/")


def is_master_ref(path: str | Path) -> bool:
    """True only for a file path whose final component is master.png."""
    text = _posix(path)
    if not text or text.endswith("/"):
        return False
    name = PurePosixPath(text).name
    return name == MASTER_FILENAME


def assert_character_refs(refs: Iterable[str | Path] | None) -> list[str]:
    """Normalize and validate B-tier production refs.

    Accepts only ``…/master.png`` (relative or absolute). Rejects directories,
    sheet/turnaround/face files, and empty tokens.
    """
    out: list[str] = []
    for raw in refs or []:
        text = _posix(raw)
        if not text:
            continue
        if text.endswith("/") or PurePosixPath(text).suffix == "":
            # bare dir or extension-less token — workers must not scan dirs
            raise CharacterRefError(
                f"character_refs must be {MASTER_FILENAME} files, not directories: {text!r}"
            )
        if not is_master_ref(text):
            raise CharacterRefError(
                f"character_refs must end with /{MASTER_FILENAME}, got {text!r}"
            )
        out.append(text)
    return out


def character_master_relpath(character_id: str) -> str:
    cid = str(character_id or "").strip()
    if not cid:
        raise CharacterRefError("character_id required for master path")
    return f"characters/{cid}/{MASTER_FILENAME}"


@dataclass
class AssetRef:
    """Locked asset entry in the registry (character / scene / prop)."""

    asset_id: str
    kind: str  # character | scene | prop
    master_path: str = ""
    sheet_paths: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> AssetRef:
        obj = dict(raw or {})
        sheets = [str(x) for x in (obj.get("sheet_paths") or obj.get("sheet_refs") or []) if x]
        master = str(obj.get("master_path") or obj.get("master") or "").strip()
        if master and not is_master_ref(master):
            raise CharacterRefError(
                f"AssetRef.master_path must be {MASTER_FILENAME}, got {master!r}"
            )
        return cls(
            asset_id=str(obj.get("asset_id") or obj.get("id") or ""),
            kind=str(obj.get("kind") or "character"),
            master_path=master,
            sheet_paths=sheets,
            meta=dict(obj.get("meta") or {}),
        )


@dataclass
class ShotContract:
    """Shot planner output: tier + prompts + production refs."""

    shot_id: str
    tier: ShotTier = "A"
    prompt: str = ""
    negative_prompt: str = ""
    character_refs: list[str] = field(default_factory=list)
    sheet_refs: list[str] = field(default_factory=list)
    control_mode: ControlMode = "none"
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.character_refs = assert_character_refs(self.character_refs)
        self.sheet_refs = [_posix(x) for x in self.sheet_refs if _posix(x)]
        tier = str(self.tier or "A").strip().upper()
        if tier not in {"A", "B", "C"}:
            raise ValueError(f"shot tier must be A|B|C, got {self.tier!r}")
        self.tier = tier  # type: ignore[assignment]
        mode = str(self.control_mode or "none").strip().lower()
        if mode not in {"identity", "pose", "depth", "compose", "none"}:
            raise ValueError(f"control_mode invalid: {self.control_mode!r}")
        self.control_mode = mode  # type: ignore[assignment]
        if self.tier == "B" and self.control_mode == "none":
            self.control_mode = "identity"
        if self.tier == "B" and not self.character_refs:
            raise CharacterRefError("tier B shots require at least one master.png character_ref")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ShotContract:
        obj = dict(raw or {})
        refs = obj.get("character_refs") or obj.get("character_ref") or []
        if isinstance(refs, (str, Path)):
            refs = [refs]
        sheets = obj.get("sheet_refs") or []
        if isinstance(sheets, (str, Path)):
            sheets = [sheets]
        seed = obj.get("seed")
        try:
            seed_i = int(seed) if seed is not None and str(seed).strip() != "" else None
        except (TypeError, ValueError):
            seed_i = None
        return cls(
            shot_id=str(obj.get("shot_id") or obj.get("id") or ""),
            tier=str(obj.get("tier") or "A"),  # type: ignore[arg-type]
            prompt=str(obj.get("prompt") or ""),
            negative_prompt=str(obj.get("negative_prompt") or ""),
            character_refs=list(refs),
            sheet_refs=list(sheets),
            control_mode=str(obj.get("control_mode") or "none"),  # type: ignore[arg-type]
            seed=seed_i,
            meta=dict(obj.get("meta") or {}),
        )


@dataclass
class QcReport:
    """Keyframe QC result. PASS is required before video_backend.generate()."""

    verdict: QcVerdict = "fail"
    layer: str = "clip"  # clip | vlm | combined
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> QcReport:
        obj = dict(raw or {})
        verdict = str(obj.get("verdict") or "fail").strip().lower()
        if verdict not in {"pass", "fail", "skip"}:
            verdict = "fail"
        scores_raw = obj.get("scores") or {}
        scores: dict[str, float] = {}
        if isinstance(scores_raw, Mapping):
            for k, v in scores_raw.items():
                try:
                    scores[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        return cls(
            verdict=verdict,  # type: ignore[arg-type]
            layer=str(obj.get("layer") or "clip"),
            reasons=[str(x) for x in (obj.get("reasons") or [])],
            scores=scores,
            meta=dict(obj.get("meta") or {}),
        )


@dataclass
class KeyframeContract:
    """Generated composition still + QC state. Video consumes keyframe_final only."""

    shot_id: str
    path: str = ""
    final_path: str = ""
    status: KeyframeStatus = "pending"
    qc: QcReport | None = None
    backend: str = ""
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.final_path and self.path:
            parent = PurePosixPath(_posix(self.path)).parent
            self.final_path = str(parent / KEYFRAME_FINAL_NAME) if str(parent) != "." else KEYFRAME_FINAL_NAME

    @property
    def qc_passed(self) -> bool:
        return bool(self.qc and self.qc.passed) or self.status == "qc_pass"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> KeyframeContract:
        obj = dict(raw or {})
        qc_raw = obj.get("qc")
        qc = QcReport.from_dict(qc_raw) if isinstance(qc_raw, Mapping) else None
        seed = obj.get("seed")
        try:
            seed_i = int(seed) if seed is not None and str(seed).strip() != "" else None
        except (TypeError, ValueError):
            seed_i = None
        return cls(
            shot_id=str(obj.get("shot_id") or obj.get("id") or ""),
            path=str(obj.get("path") or ""),
            final_path=str(obj.get("final_path") or ""),
            status=str(obj.get("status") or "pending"),  # type: ignore[arg-type]
            qc=qc,
            backend=str(obj.get("backend") or ""),
            seed=seed_i,
            meta=dict(obj.get("meta") or {}),
        )


@dataclass
class VideoContract:
    """Video job bound to a QC-passed keyframe_final.png."""

    shot_id: str
    keyframe_final: str = ""
    path: str = ""
    status: VideoStatus = "pending"
    backend: str = ""
    rife: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> VideoContract:
        obj = dict(raw or {})
        return cls(
            shot_id=str(obj.get("shot_id") or obj.get("id") or ""),
            keyframe_final=str(obj.get("keyframe_final") or obj.get("first_frame") or ""),
            path=str(obj.get("path") or ""),
            status=str(obj.get("status") or "pending"),  # type: ignore[arg-type]
            backend=str(obj.get("backend") or ""),
            rife=bool(obj.get("rife")),
            meta=dict(obj.get("meta") or {}),
        )


def assert_keyframe_allows_video(keyframe: KeyframeContract | Mapping[str, Any] | None) -> KeyframeContract:
    """Hard gate: FAIL / missing QC blocks video GPU time."""
    kf = keyframe if isinstance(keyframe, KeyframeContract) else KeyframeContract.from_dict(keyframe)
    if not kf.qc_passed:
        reasons = (kf.qc.reasons if kf.qc else []) or ["qc_missing_or_fail"]
        raise KeyframeQcGateError(
            f"keyframe QC gate blocked video for shot {kf.shot_id!r}: {', '.join(reasons)}"
        )
    final = _posix(kf.final_path or kf.path)
    if not final or PurePosixPath(final).name not in {KEYFRAME_FINAL_NAME, "f1.png"}:
        # Allow legacy f1.png as interim path; prefer keyframe_final.png.
        if not final:
            raise KeyframeQcGateError(f"keyframe final path missing for shot {kf.shot_id!r}")
    return kf


def b_tier_control_payload(shot: ShotContract | Mapping[str, Any]) -> dict[str, Any]:
    """Canonical B-tier JSON contract from the stills strategy plan."""
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    return {
        "character_refs": list(s.character_refs),
        "sheet_refs": list(s.sheet_refs),
        "control_mode": s.control_mode if s.control_mode != "none" else "identity",
    }


def refs_exclude_sheets(character_refs: Sequence[str], sheet_refs: Sequence[str]) -> list[str]:
    """Production refs only — drop any path that also appears as a sheet."""
    sheets = {_posix(x) for x in sheet_refs}
    return [r for r in assert_character_refs(character_refs) if r not in sheets]
