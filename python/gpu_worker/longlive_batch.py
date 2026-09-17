"""Build the NVFP4 S2 pipeline once, then sample every take with that same object.

``model_load_count`` counts real ``CausalDiffusionInferencePipeline`` constructions that
were handed to ``setup_nvfp4_pipeline``; importing a class is not a load. There is no
``inference.py`` subprocess anywhere on this path, so weights are never re-read per take.
Outputs are written straight to the caller's ``dests[take_id]`` — no newest-file guessing.
Missing output fails closed; never write a placeholder mp4.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from gpu_worker.longlive import (
    FAIL_CLOSED,
    SETUP_HINT,
    _assert_nvfp4_generator,
    _ensure_wan_layout,
    _install_torch_mmap_sitecustomize,
    batch_longlive_mode,
    longlive_num_output_frames,
    missing_longlive_requirements,
    select_longlive_mode,
    write_inference_yaml,
    write_prompt_job,
)

ProgressCallback = Callable[[str], None]
InferHook = Callable[..., Any]
INSTRUMENT_NAME = "longlive_batch.json"
# Frames the batch config is built with. Per-take duration overrides
# config.num_output_frames before each sample; the pipeline itself only pins the
# latent H/W, block size and attention window, which are identical for every take.
DEFAULT_BATCH_SECONDS = 8.0


def _skip_weights() -> bool:
    return os.environ.get("AF_SKIP_WEIGHTS", "").strip().lower() in {"1", "true", "yes", "on"}


class LongLiveBatchRunner:
    """Owns one resident NVFP4 pipeline for the lifetime of a batch."""

    def __init__(self) -> None:
        self.model_load_count = 0
        self.inference_calls = 0
        self.h3_downloads = 0
        self.pipeline: Any = None
        self.api: Any = None
        self.config: Any = None
        self.device: Any = None
        self.mode: str | None = None
        self.batch_dir: Path | None = None

    def load_model(self, *, takes: list[dict[str, Any]], workdir: Path) -> Any:
        """Construct + set up the official pipeline exactly once."""
        if self.pipeline is not None:
            return self.pipeline
        if _skip_weights():
            raise RuntimeError(f"{FAIL_CLOSED}: AF_SKIP_WEIGHTS forbids loading the NVFP4 pipeline")
        missing = missing_longlive_requirements()
        if missing:
            raise RuntimeError(
                f"{FAIL_CLOSED}: longlive not ready: " + "; ".join(missing) + ". " + SETUP_HINT
            )
        _assert_nvfp4_generator()
        _ensure_wan_layout()
        _install_torch_mmap_sitecustomize()
        from gpu_worker import longlive_pipeline
        from gpu_worker.stack import stop_comfy_for_longlive

        # The generator has to land on a card Comfy is not already holding.
        stop_comfy_for_longlive()

        self.mode = batch_longlive_mode(takes)
        self.batch_dir = Path(workdir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = write_inference_yaml(
            self.batch_dir / "inference.yaml",
            data_path=self.batch_dir / "data",
            output_folder=self.batch_dir / "out",
            seconds=DEFAULT_BATCH_SECONDS,
            seed=int(takes[0].get("seed") or 11),
            i2v=self.mode == "i2v",
        )
        api = longlive_pipeline.import_official_api()
        config = longlive_pipeline.load_official_config(api, yaml_path)
        pipeline, device = longlive_pipeline.build_pipeline(api, config)
        if pipeline is None or isinstance(pipeline, type):
            raise RuntimeError(
                f"{FAIL_CLOSED}: build_pipeline returned {pipeline!r}; expected an "
                "initialized CausalDiffusionInferencePipeline instance"
            )
        self.api = api
        self.config = config
        self.pipeline = pipeline
        self.device = device
        self.model_load_count += 1
        return self.pipeline

    def sample(
        self,
        *,
        take: dict[str, Any],
        dest: Path,
        data_path: Path,
        frames: int,
        mode: str,
        **_: Any,
    ) -> Path:
        """Production sampler: one ``pipeline.inference`` call on the resident object."""
        if self.pipeline is None:
            raise RuntimeError(f"{FAIL_CLOSED}: sample() called before the pipeline was built")
        from gpu_worker import longlive_pipeline

        caption = ""
        if mode != "i2v":
            caption = Path(data_path).read_text(encoding="utf-8").strip()
        self.config.data_path = str(data_path)
        produced = longlive_pipeline.sample_take(
            self.api,
            self.pipeline,
            self.config,
            self.device,
            dest=Path(dest),
            frames=int(frames),
            seed=int(take.get("seed") or 11),
            i2v=mode == "i2v",
            caption=caption,
            data_path=Path(data_path) if mode == "i2v" else None,
        )
        self.inference_calls += 1
        return produced

    def release_model(self) -> None:
        """Release LongLive VRAM before a separate MOSS-SFX process may start."""
        pipeline = self.pipeline
        self.pipeline = None
        self.api = None
        self.config = None
        self.device = None
        del pipeline
        gc.collect()
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
                cuda.empty_cache()
        except (AttributeError, ImportError, RuntimeError):
            # CPU contract tests do not install torch; a torn-down CUDA context
            # can also reject cleanup after an earlier fail-closed inference.
            pass

    def infer_take(
        self,
        take: dict[str, Any],
        dest: Path,
        *,
        run: InferHook | None = None,
    ) -> dict[str, Any]:
        """Stage this take's conditioning, then sample it with the resident pipeline."""
        if self.pipeline is None:
            raise RuntimeError(f"{FAIL_CLOSED}: infer_take() called before the pipeline was built")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        take_id = str(take.get("take_id") or take.get("id") or "take")
        # Staged lazily: a same-scene continuation take only gets its conditioning
        # still once the preceding take has rendered and its last frame was extracted.
        mode = select_longlive_mode(take)
        if mode != self.mode:
            raise RuntimeError(
                f"{FAIL_CLOSED}: take {take_id} resolved to {mode} but the resident pipeline "
                f"was built for {self.mode}. Expected conditioning still at "
                f"{take.get('first_frame_path') or '<unset>'}; refusing to downgrade."
            )
        workdir = dest.parent / f".longlive-batch-{take_id}"
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)
        data_path = write_prompt_job(workdir / "data", take, i2v=mode == "i2v")
        frames = longlive_num_output_frames(float(take.get("duration") or DEFAULT_BATCH_SECONDS))
        sampler = run or self.sample
        produced = sampler(
            take=take,
            dest=dest,
            data_path=data_path,
            frames=frames,
            mode=mode,
            pipeline=self.pipeline,
            runner=self,
        )
        src = Path(produced) if produced else dest
        if not src.is_file() or src.stat().st_size < 32:
            raise RuntimeError(
                f"{FAIL_CLOSED}: longlive produced no video for {take_id}. {SETUP_HINT}"
            )
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        if dest.stat().st_size < 32:
            raise RuntimeError(f"{FAIL_CLOSED}: longlive wrote empty video for {take_id}")
        return {
            "take_id": take_id,
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "mode": mode,
            "frames": frames,
        }


def sample_with_loaded_pipeline(*, runner: LongLiveBatchRunner, **kwargs: Any) -> Path:
    """Default infer hook. Delegates to the runner that owns the resident pipeline."""
    if _skip_weights():
        raise RuntimeError(f"{FAIL_CLOSED}: AF_SKIP_WEIGHTS forbids placeholder media")
    return runner.sample(**kwargs)


def submit_longlive_batch(
    takes: list[dict[str, Any]],
    dests: dict[str, Path],
    *,
    progress: ProgressCallback | None = None,
    infer: InferHook | None = None,
    runner: LongLiveBatchRunner | None = None,
) -> dict[str, Any]:
    """Run every take through one NVFP4 S2 load. Map outputs explicitly by take_id."""
    if not takes:
        raise RuntimeError(f"{FAIL_CLOSED}: longlive batch is empty")
    take_ids: list[str] = []
    for take in takes:
        take_id = str(take.get("take_id") or take.get("id") or "")
        if not take_id:
            raise RuntimeError(f"{FAIL_CLOSED}: take missing take_id")
        if take_id in take_ids:
            raise RuntimeError(f"{FAIL_CLOSED}: duplicate take_id={take_id} in batch")
        if take_id not in dests:
            raise RuntimeError(f"{FAIL_CLOSED}: no dest mapped for take_id={take_id}")
        take_ids.append(take_id)
    resolved = {str(Path(dests[tid]).resolve()) for tid in take_ids}
    if len(resolved) != len(take_ids):
        raise RuntimeError(f"{FAIL_CLOSED}: two takes map to the same dest; mapping is ambiguous")

    session = runner or LongLiveBatchRunner()
    first_dest = Path(dests[take_ids[0]])
    loaded_pipeline: Any = None
    try:
        session.load_model(takes=takes, workdir=first_dest.parent / ".longlive-batch")
        if session.model_load_count != 1:
            raise RuntimeError(
                f"{FAIL_CLOSED}: expected model_load_count==1, got {session.model_load_count}"
            )
        loaded_pipeline = session.pipeline

        mapped: dict[str, dict[str, Any]] = {}
        outputs: dict[str, str] = {}
        for take, take_id in zip(takes, take_ids):
            if progress:
                progress(f"anim:{take_id}")
            result = session.infer_take(take, Path(dests[take_id]), run=infer)
            mapped[take_id] = result
            outputs[take_id] = result["path"]
            if session.pipeline is not loaded_pipeline:
                raise RuntimeError(f"{FAIL_CLOSED}: pipeline object changed during take {take_id}")
        if session.model_load_count != 1:
            raise RuntimeError(
                f"{FAIL_CLOSED}: model reloaded during batch ({session.model_load_count})"
            )
        instrument = {
            "model_load_count": session.model_load_count,
            "inference_calls": session.inference_calls,
            "h3_downloads": session.h3_downloads,
            "mode": session.mode,
            "take_ids": take_ids,
            "outputs": outputs,
            "native_audio": False,
            "backend": "longlive",
        }
        marker = first_dest.parent / INSTRUMENT_NAME
        marker.write_text(json.dumps(instrument, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"longlive_batch": instrument}, ensure_ascii=False), flush=True)
        return {"ok": True, "mapped": mapped, **instrument}
    finally:
        loaded_pipeline = None
        session.release_model()
