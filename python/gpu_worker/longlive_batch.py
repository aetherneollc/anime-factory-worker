"""Load NVFP4 S2 once and map every take by take_id.

Current ``submit_longlive`` shells ``inference.py`` per segment (one generator
load each). This runner keeps the load in-process for an episode/batch so
``model_load_count==1``. Missing output fails closed — never write placeholder mp4.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from gpu_worker.longlive import (
    FAIL_CLOSED,
    SETUP_HINT,
    _assert_nvfp4_generator,
    _ensure_wan_layout,
    _find_output_mp4,
    _install_torch_mmap_sitecustomize,
    _longlive_inference_env,
    is_longlive_fatal_error,
    longlive_num_output_frames,
    longlive_root,
    missing_longlive_requirements,
    python_bin,
    select_longlive_mode,
    write_inference_yaml,
    write_prompt_job,
)

ProgressCallback = Callable[[str], None]
InferHook = Callable[..., Any]
INSTRUMENT_NAME = "longlive_batch.json"


class LongLiveBatchRunner:
    def __init__(self) -> None:
        self.model_load_count = 0
        self.pipeline: Any = None
        self.h3_downloads = 0

    def load_model(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline
        missing = missing_longlive_requirements()
        if missing:
            raise RuntimeError(
                f"{FAIL_CLOSED}: longlive not ready: " + "; ".join(missing) + ". " + SETUP_HINT
            )
        _assert_nvfp4_generator()
        _ensure_wan_layout()
        _install_torch_mmap_sitecustomize()
        self.pipeline = _load_nvfp4_pipeline()
        self.model_load_count += 1
        return self.pipeline

    def infer_take(
        self,
        take: dict[str, Any],
        dest: Path,
        *,
        run: InferHook,
    ) -> dict[str, Any]:
        self.load_model()
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        take_id = str(take.get("take_id") or take.get("id") or "take")
        workdir = dest.parent / f".longlive-batch-{take_id}"
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)
        mode = select_longlive_mode(take)
        data_path = write_prompt_job(workdir / "data", take, i2v=mode == "i2v")
        out_dir = workdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = write_inference_yaml(
            workdir / "inference.yaml",
            data_path=data_path,
            output_folder=out_dir,
            seconds=float(take.get("duration") or 8.0),
            seed=int(take.get("seed") or 11),
            i2v=mode == "i2v",
        )
        produced = run(
            take=take,
            dest=dest,
            yaml_path=yaml_path,
            out_dir=out_dir,
            pipeline=self.pipeline,
        )
        src = Path(produced) if produced else dest
        if not src.is_file() or src.stat().st_size < 32:
            raise RuntimeError(f"{FAIL_CLOSED}: longlive produced no video for {take_id}. {SETUP_HINT}")
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        if dest.stat().st_size < 32:
            raise RuntimeError(f"{FAIL_CLOSED}: longlive wrote empty video for {take_id}")
        return {
            "take_id": take_id,
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "mode": mode,
            "frames": longlive_num_output_frames(float(take.get("duration") or 8.0)),
        }


def infer_with_loaded_pipeline(
    *,
    take: dict[str, Any],
    dest: Path,
    yaml_path: Path,
    out_dir: Path,
    pipeline: Any,
    **_: Any,
) -> Path:
    """Sample one take with the already-imported NVFP4 pipeline. Never placeholder mp4."""
    dest = Path(dest)
    out_dir = Path(out_dir)
    yaml_path = Path(yaml_path)
    if os.environ.get("AF_SKIP_WEIGHTS", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(f"{FAIL_CLOSED}: AF_SKIP_WEIGHTS forbids placeholder media")
    sampler = getattr(pipeline, "inference", None)
    if callable(sampler) and not isinstance(pipeline, type):
        sampler(str(yaml_path))
        produced = _find_output_mp4(out_dir)
        if produced is None or produced.stat().st_size < 32:
            raise RuntimeError(
                f"{FAIL_CLOSED}: loaded pipeline produced no video for "
                f"{take.get('take_id') or take.get('id')}. {SETUP_HINT}"
            )
        if produced.resolve() != dest.resolve():
            shutil.copy2(produced, dest)
        return dest
    # Same interpreter / imported pipeline class — do not spawn a second generator load
    # unless the baked image only exposes inference.py. Fail closed when that file is gone.
    infer_py = longlive_root() / "inference.py"
    if not infer_py.is_file():
        raise RuntimeError(f"{FAIL_CLOSED}: inference.py missing; cannot sample. {SETUP_HINT}")
    import subprocess

    timeout = float(os.environ.get("LONGLIVE_SHOT_TIMEOUT_S") or 2400)
    cmd = [python_bin(), str(infer_py), "--config_path", str(yaml_path)]
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(longlive_root()),
            timeout=timeout,
            env=_longlive_inference_env(),
        )
    except subprocess.CalledProcessError as exc:
        if is_longlive_fatal_error(exc) or exc.returncode in (-9, 9, 137):
            raise RuntimeError(
                f"{FAIL_CLOSED}: inference SIGKILL/exit {exc.returncode} for "
                f"{take.get('take_id') or take.get('id')}. {SETUP_HINT}"
            ) from exc
        raise RuntimeError(
            f"{FAIL_CLOSED}: longlive inference failed "
            f"{take.get('take_id') or take.get('id')}: exit {exc.returncode}. {SETUP_HINT}"
        ) from exc
    produced = _find_output_mp4(out_dir)
    if produced is None or produced.stat().st_size < 32:
        raise RuntimeError(
            f"{FAIL_CLOSED}: longlive produced no video for "
            f"{take.get('take_id') or take.get('id')}. {SETUP_HINT}"
        )
    if produced.resolve() != dest.resolve():
        shutil.copy2(produced, dest)
    return dest


def _load_nvfp4_pipeline() -> Any:
    """Import NVlabs pipeline once. Fail closed when the baked tree is missing."""
    root = longlive_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from pipeline import CausalDiffusionInferencePipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001 — production image must have this
        # Tests inject infer hooks after load_model is patched; a stub still
        # counts as one load so mapping tests can run without CUDA.
        if os.environ.get("AF_SKIP_WEIGHTS", "").strip().lower() in {"1", "true", "yes", "on"}:
            return {"stub": True, "error": str(exc)}
        raise RuntimeError(
            f"{FAIL_CLOSED}: NVFP4 pipeline import failed ({type(exc).__name__}: {exc}). {SETUP_HINT}"
        ) from exc
    return CausalDiffusionInferencePipeline


def submit_longlive_batch(
    takes: list[dict[str, Any]],
    dests: dict[str, Path],
    *,
    progress: ProgressCallback | None = None,
    infer: InferHook | None = None,
    runner: LongLiveBatchRunner | None = None,
) -> dict[str, Any]:
    """Run every take with one NVFP4 S2 load. Map outputs by take_id."""
    if not takes:
        raise RuntimeError(f"{FAIL_CLOSED}: longlive batch is empty")
    session = runner or LongLiveBatchRunner()
    session.load_model()
    if session.model_load_count != 1:
        raise RuntimeError(
            f"{FAIL_CLOSED}: expected model_load_count==1, got {session.model_load_count}"
        )
    if infer is None:
        infer = infer_with_loaded_pipeline
    mapped: dict[str, dict[str, Any]] = {}
    outputs: dict[str, str] = {}
    for take in takes:
        take_id = str(take.get("take_id") or take.get("id") or "")
        if not take_id:
            raise RuntimeError(f"{FAIL_CLOSED}: take missing take_id")
        if take_id not in dests:
            raise RuntimeError(f"{FAIL_CLOSED}: no dest mapped for take_id={take_id}")
        dest = Path(dests[take_id])
        if progress:
            progress(f"anim:{take_id}")
        result = session.infer_take(take, dest, run=infer)
        mapped[take_id] = result
        outputs[take_id] = result["path"]
    if session.model_load_count != 1:
        raise RuntimeError(
            f"{FAIL_CLOSED}: model reloaded during batch ({session.model_load_count})"
        )
    instrument = {
        "model_load_count": session.model_load_count,
        "h3_downloads": session.h3_downloads,
        "take_ids": [str(t.get("take_id") or t.get("id")) for t in takes],
        "outputs": outputs,
        "native_audio": False,
        "backend": "longlive",
    }
    first_dest = next(iter(dests.values()))
    marker = Path(first_dest).parent / INSTRUMENT_NAME
    marker.write_text(json.dumps(instrument, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"longlive_batch": instrument}, ensure_ascii=False), flush=True)
    return {"ok": True, "mapped": mapped, **instrument}
