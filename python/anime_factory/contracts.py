"""Stable Asset / Shot / Keyframe / Video / QC contracts for the stills pipeline.

The story layer does not cap how many references a shot may carry. A video
backend may truncate for its own model; that limit does not live here.

B-tier hard rule: ``character_refs`` are production Master paths only
(``…/master.png``). Sheets stay in ``sheet_refs`` (docs / registry) and must
not be fed into Control or Video by default. Workers must not scan a whole
character directory.

Video consumes a QC-passed reference pack (``references: ReferenceAsset[]``)
plus prompt, camera, motion, duration, resolution, and seed. Character images
in that pack must be ``master.png``. The preview keyframe is for QC and
review; it is not a video reference and not a forced first frame.
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
    scene_refs: list[str] = field(default_factory=list)
    control_mode: ControlMode = "none"
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.character_refs = assert_character_refs(self.character_refs)
        self.sheet_refs = [_posix(x) for x in self.sheet_refs if _posix(x)]
        self.scene_refs = [_posix(x) for x in self.scene_refs if _posix(x)]
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
        scenes = obj.get("scene_refs") or obj.get("scene_ref") or []
        if isinstance(scenes, (str, Path)):
            scenes = [scenes]
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
            scene_refs=list(scenes),
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
    """Generated composition still + QC state. Review only; not a video reference."""

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
    """Video job. ``keyframe_final`` is a review pointer, not a reference image."""

    shot_id: str
    prompt: str = ""
    camera: str = ""
    motion: str = ""
    duration: float = 5.0
    resolution: str = "720P"
    seed: int | None = None
    references: list[str] = field(default_factory=list)
    keyframe_final: str = ""
    path: str = ""
    status: VideoStatus = "pending"
    backend: str = ""
    rife: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.references = [_posix(x) for x in self.references if _posix(x)]
        try:
            self.duration = float(self.duration)
        except (TypeError, ValueError):
            self.duration = 5.0
        if self.seed is not None:
            try:
                self.seed = int(self.seed)
            except (TypeError, ValueError):
                self.seed = None

    @property
    def ref_images(self) -> list[str]:
        return list(self.references)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ref_images"] = list(self.references)
        data["images"] = list(self.references)
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> VideoContract:
        obj = dict(raw or {})
        refs = obj.get("references")
        if refs is None:
            refs = obj.get("ref_images") or obj.get("images") or []
        seed = obj.get("seed")
        try:
            seed_i = int(seed) if seed is not None and str(seed).strip() != "" else None
        except (TypeError, ValueError):
            seed_i = None
        try:
            duration = float(obj.get("duration") if obj.get("duration") is not None else obj.get("duration_s") or 5.0)
        except (TypeError, ValueError):
            duration = 5.0
        return cls(
            shot_id=str(obj.get("shot_id") or obj.get("id") or ""),
            prompt=str(obj.get("prompt") or ""),
            camera=str(obj.get("camera") or ""),
            motion=str(obj.get("motion") or ""),
            duration=duration,
            resolution=str(obj.get("resolution") or "720P"),
            seed=seed_i,
            references=[_posix(x) for x in refs if _posix(str(x) if not isinstance(x, Mapping) else x.get("path") or "")],
            keyframe_final=str(obj.get("keyframe_final") or ""),
            path=str(obj.get("path") or ""),
            status=str(obj.get("status") or "pending"),  # type: ignore[arg-type]
            backend=str(obj.get("backend") or ""),
            rife=bool(obj.get("rife")),
            meta=dict(obj.get("meta") or {}),
        )


REF_PACK_ROLES = ("character", "scene", "prop", "preview")
RefPackQcStatus = Literal["pass", "fail", "skip", "pending"]


class RefPackQcGateError(KeyframeQcGateError):
    """Video must not run unless the ref pack passed QC."""


@dataclass
class ReferenceAsset:
    """One story-layer reference. Character slots are master.png only.

    The story layer does not cap how many of these a pack may hold.
    """

    path: str
    role: str = "character"
    qc_status: RefPackQcStatus = "pending"
    asset_id: str = ""

    def __post_init__(self) -> None:
        self.path = _posix(self.path)
        self.asset_id = str(self.asset_id or "").strip()
        role = str(self.role or "").strip().lower()
        if role not in REF_PACK_ROLES:
            raise ValueError(f"reference role must be {REF_PACK_ROLES}, got {self.role!r}")
        self.role = role
        status = str(self.qc_status or "pending").strip().lower()
        if status not in {"pass", "fail", "skip", "pending"}:
            status = "fail"
        self.qc_status = status  # type: ignore[assignment]
        if self.role == "character" and not is_master_ref(self.path):
            raise CharacterRefError(
                f"reference character image must be {MASTER_FILENAME}, got {self.path!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ReferenceAsset:
        obj = dict(raw or {})
        return cls(
            path=str(obj.get("path") or ""),
            role=str(obj.get("role") or "character"),
            qc_status=str(obj.get("qc_status") or obj.get("qc") or "pending"),  # type: ignore[arg-type]
            asset_id=str(obj.get("asset_id") or obj.get("id") or ""),
        )


# Compatibility alias while call sites migrate off the old image record name.
RefPackImage = ReferenceAsset


def _as_reference(item: Any) -> ReferenceAsset:
    if isinstance(item, ReferenceAsset):
        return item
    if isinstance(item, Mapping):
        return ReferenceAsset.from_dict(item)
    raise CharacterRefError(f"reference must be a path record, got {item!r}")


@dataclass(init=False)
class RefPackContract:
    """QC'd reference pack. ``references`` is unbounded at this layer.

    Character images are ``master.png``. ``images`` is a compatibility alias
    for ``references``. A preview keyframe path is review metadata only.
    """

    shot_id: str
    references: list[ReferenceAsset] = field(default_factory=list)
    qc_status: RefPackQcStatus = "fail"
    preview_path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        shot_id: str,
        references: Sequence[Any] | None = None,
        qc_status: str = "fail",
        preview_path: str = "",
        meta: Mapping[str, Any] | None = None,
        images: Sequence[Any] | None = None,
    ) -> None:
        chosen = references if references is not None else images
        self.shot_id = str(shot_id or "")
        self.references = [_as_reference(item) for item in (chosen or [])]
        self.qc_status = str(qc_status or "fail")  # type: ignore[assignment]
        self.preview_path = preview_path
        self.meta = dict(meta or {})
        self.__post_init__()

    def __post_init__(self) -> None:
        status = str(self.qc_status or "fail").strip().lower()
        if status not in {"pass", "fail", "skip", "pending"}:
            status = "fail"
        self.qc_status = status  # type: ignore[assignment]
        self.preview_path = _posix(self.preview_path)

    @property
    def images(self) -> list[ReferenceAsset]:
        """Compatibility alias for ``references``."""
        return self.references

    @images.setter
    def images(self, value: Sequence[Any]) -> None:
        self.references = [_as_reference(item) for item in (value or [])]

    def to_dict(self) -> dict[str, Any]:
        data = {
            "shot_id": self.shot_id,
            "references": [item.to_dict() for item in self.references],
            "qc_status": self.qc_status,
            "preview_path": self.preview_path,
            "meta": dict(self.meta),
        }
        data["images"] = list(data["references"])
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> RefPackContract:
        obj = dict(raw or {})
        refs = obj.get("references")
        if refs is None:
            refs = obj.get("images") or []
        return cls(
            shot_id=str(obj.get("shot_id") or obj.get("id") or ""),
            references=list(refs),
            qc_status=str(obj.get("qc_status") or "fail"),
            preview_path=str(obj.get("preview_path") or ""),
            meta=dict(obj.get("meta") or {}),
        )


@dataclass
class VideoRequest:
    """Story-layer video request. Reference count is not capped here."""

    shot_id: str
    references: list[ReferenceAsset] = field(default_factory=list)
    prompt: str = ""
    camera: str = ""
    motion: str = ""
    duration: float = 5.0
    resolution: str = "720P"
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: list[ReferenceAsset] = []
        for item in self.references:
            normalized.append(_as_reference(item))
        self.references = normalized
        try:
            self.duration = float(self.duration)
        except (TypeError, ValueError):
            self.duration = 5.0
        if self.seed is not None:
            try:
                self.seed = int(self.seed)
            except (TypeError, ValueError):
                self.seed = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["references"] = [item.to_dict() for item in self.references]
        return data


def assert_ref_pack_allows_video(
    pack: RefPackContract | Mapping[str, Any] | None,
) -> RefPackContract:
    """Hard gate: no video GPU unless the reference pack QC passed.

    Does not cap reference count. Character slots must be ``master.png`` and
    themselves QC-passed. An empty pack is not a count error here; the video
    backend refuses to invent a preview frame.
    """
    ref = pack if isinstance(pack, RefPackContract) else RefPackContract.from_dict(pack)
    if ref.qc_status != "pass":
        raise RefPackQcGateError(
            f"ref pack QC gate blocked video for shot {ref.shot_id!r}: qc_status={ref.qc_status!r}"
        )
    for image in ref.references:
        if image.role == "character":
            if not is_master_ref(image.path):
                raise CharacterRefError(
                    f"ref pack character image must be {MASTER_FILENAME}, got {image.path!r}"
                )
            if image.qc_status != "pass":
                raise RefPackQcGateError(
                    f"character master {image.path!r} qc_status={image.qc_status!r} blocks video"
                )
    return ref


def ref_pack_from_keyframe(keyframe: KeyframeContract | Mapping[str, Any] | None) -> RefPackContract:
    """QC status from a keyframe. The still is not copied into ``references``."""
    kf = keyframe if isinstance(keyframe, KeyframeContract) else KeyframeContract.from_dict(keyframe)
    final = _posix(kf.final_path or kf.path)
    status: RefPackQcStatus = "pass" if kf.qc_passed else "fail"
    return RefPackContract(
        shot_id=kf.shot_id,
        references=[],
        qc_status=status,
        preview_path=final,
        meta=dict(kf.meta),
    )


def assert_keyframe_allows_video(keyframe: KeyframeContract | Mapping[str, Any] | None) -> KeyframeContract:
    """QC gate for a keyframe. Does not turn the preview into a video reference."""
    kf = keyframe if isinstance(keyframe, KeyframeContract) else KeyframeContract.from_dict(keyframe)
    assert_ref_pack_allows_video(ref_pack_from_keyframe(kf))
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
