"""Stand-ins for the NVlabs modules the NVFP4 path imports out of LONGLIVE_ROOT.

Only CUDA and the upstream tree are faked. ``gpu_worker.longlive*`` runs for real, so
these fakes record exactly how many times the production code constructs a pipeline,
sets it up, and samples with it.

The shapes and call signatures mirror NVlabs/LongLive at LONGLIVE_REF:
``inference.py``, ``utils/inference_utils.py`` and ``utils/dataset.py``.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class FakeTensor:
    def __init__(self, shape, dtype=None, device=None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.device = device

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _derive(self, shape, **over) -> "FakeTensor":
        return FakeTensor(
            shape,
            dtype=over.get("dtype", self.dtype),
            device=over.get("device", self.device),
        )

    def unsqueeze(self, dim: int) -> "FakeTensor":
        shape = list(self.shape)
        shape.insert(dim if dim >= 0 else len(shape) + dim + 1, 1)
        return self._derive(shape)

    def to(self, *_args, **kwargs) -> "FakeTensor":
        return self._derive(self.shape, **kwargs)

    def repeat(self, *sizes) -> "FakeTensor":
        return self._derive([d * s for d, s in zip(self.shape, sizes)])

    def __getitem__(self, _idx) -> "FakeTensor":
        return self._derive(self.shape[1:])


class FakeTorch(types.ModuleType):
    bfloat16 = "bfloat16"

    def __init__(self, log: dict[str, Any]):
        super().__init__("torch")
        self.log = log

    def device(self, spec):
        return f"device({spec})"

    def set_grad_enabled(self, flag):
        self.log.setdefault("grad_enabled", []).append(flag)

    def randn(self, shape, device=None, dtype=None, generator=None):
        self.log.setdefault("randn", []).append(tuple(shape))
        return FakeTensor(shape, dtype=dtype, device=device)

    @contextmanager
    def inference_mode(self):
        self.log["inference_mode_depth"] = self.log.get("inference_mode_depth", 0) + 1
        yield
        self.log["inference_mode_depth"] -= 1


class _Model:
    def __init__(self, log):
        self.log = log

    def eval(self):
        self.log.setdefault("eval", []).append(True)
        return self

    def requires_grad_(self, flag):
        self.log.setdefault("requires_grad_", []).append(flag)
        return self


class _Generator:
    def __init__(self, log):
        self.model = _Model(log)

    def to(self, **_kwargs):
        return self


class _Vae:
    def __init__(self, log):
        self.log = log
        self.model = self

    def encode_to_latent(self, pixel: FakeTensor) -> FakeTensor:
        self.log.setdefault("encode_to_latent", []).append(pixel.shape)
        # Wan 2.2 TI2V-5B: one RGB frame -> one clean latent frame.
        batch, _c, _t, height, width = pixel.shape
        return FakeTensor((batch, 1, 48, height // 16, width // 16))

    def clear_cache(self):
        self.log["clear_cache"] = self.log.get("clear_cache", 0) + 1


class _TextEncoder:
    def __init__(self, log):
        self.log = log
        self.device = "cpu"

    def to(self, device):
        self.device = str(device)
        self.log.setdefault("text_encoder_to", []).append(self.device)
        return self


class FakeCausalDiffusionInferencePipeline:
    """Counts constructions so a class import can never pass as a model load."""

    def __init__(self, config, device=None, log: dict[str, Any] | None = None):
        log = log if log is not None else {}
        self.log = log
        self.config = config
        self.device = device
        self.generator = _Generator(log)
        self.text_encoder = _TextEncoder(log)
        self.vae = _Vae(log)
        log.setdefault("constructed", []).append(id(self))
        log.setdefault("constructed_cwds", []).append(str(Path.cwd()))

    def inference(self, **kwargs):
        self.log.setdefault("inference", []).append(
            {
                "pipeline_id": id(self),
                "noise": kwargs["noise"].shape,
                "blocks": len(kwargs["text_prompts"][0]),
                "prompt": kwargs["text_prompts"][0][0],
                "initial_latent": getattr(kwargs.get("initial_latent"), "shape", None),
            }
        )
        frames = kwargs["noise"].shape[1]
        return FakeTensor((1, frames, 3, 704, 1280))


class FakeImagePromptDataset:
    """Mirrors utils/dataset.py: images/<stem>.<ext> + prompts/<stem>.txt."""

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def __init__(self, data_path, image_size, num_blocks, scene_cut_prefix="", log=None):
        self.log = log if log is not None else {}
        root = Path(data_path)
        if not root.is_dir():
            raise ValueError(f"i2v image data_path is not a directory: {root}")
        image_dir = root / "images" if (root / "images").is_dir() else root
        self.prompt_dir = root / "prompts" if (root / "prompts").is_dir() else image_dir
        found: list[Path] = []
        for ext in self.IMAGE_EXTENSIONS:
            found.extend(image_dir.glob(f"*{ext}"))
        self.images = sorted(set(found), key=lambda p: p.name)
        if not self.images:
            raise ValueError(f"No images found in {image_dir}")
        self.image_size = tuple(image_size)
        self.num_blocks = int(num_blocks)
        self.scene_cut_prefix = scene_cut_prefix
        self.log.setdefault("datasets", []).append(
            {
                "data_path": str(root),
                "image_size": self.image_size,
                "num_blocks": self.num_blocks,
                "images": [p.name for p in self.images],
            }
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        txt = self.prompt_dir / f"{image_path.stem}.txt"
        if not txt.exists():
            raise ValueError(f"No prompt file for image {image_path.name}: expected {txt}")
        shots = [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not shots:
            raise ValueError(f"Prompt file is empty: {txt}")
        base, extra = divmod(self.num_blocks, len(shots))
        prompts: list[str] = []
        for shot_idx, caption in enumerate(shots):
            for block in range(base + (1 if shot_idx < extra else 0)):
                prefix = self.scene_cut_prefix if shot_idx > 0 and block == 0 else ""
                prompts.append(prefix + caption)
        prompts = (prompts + [prompts[-1]] * self.num_blocks)[: self.num_blocks]
        # Dataset yields [C, H, W]; image_prompt_collate_fn adds the batch axis.
        return {
            "image": FakeTensor((3, self.image_size[0], self.image_size[1])),
            "prompts": prompts,
            "idx": idx,
        }


def _parse_yaml(text: str) -> dict[str, Any]:
    """Parse the flat/nested subset written by write_inference_yaml."""

    def cast(raw: str) -> Any:
        if raw in ("true", "false"):
            return raw == "true"
        if raw.startswith("'") and raw.endswith("'"):
            return raw[1:-1].replace("''", "'")
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    last_key: str | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if body.startswith("- "):
            container.setdefault(last_key, [])
            if not isinstance(container[last_key], list):
                container[last_key] = []
            container[last_key].append(cast(body[2:].strip()))
            continue
        key, _, value = body.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            container[key] = child
            stack.append((indent, child))
            last_key = key
        else:
            container[key] = cast(value)
            last_key = key
    return root


class Cfg:
    """Attribute/`get` access over a plain dict, like an OmegaConf node."""

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(name)
        value = data[name]
        return Cfg(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_data")[name] = value

    def __contains__(self, name):
        return name in object.__getattribute__(self, "_data")

    def get(self, name, default=None):
        value = object.__getattribute__(self, "_data").get(name, default)
        return Cfg(value) if isinstance(value, dict) else value

    def raw(self) -> dict[str, Any]:
        return object.__getattribute__(self, "_data")


# utils/config.py SECTION_KEYS.
_SECTION_KEYS = (
    "infra",
    "algorithm",
    "training",
    "data",
    "evaluation",
    "inference",
    "logging",
    "checkpoints",
)


def fake_normalize_config(config: Cfg) -> Cfg:
    data = config.raw()
    for section in _SECTION_KEYS:
        values = data.get(section)
        if isinstance(values, dict):
            data.update(values)
    # normalize_config also lifts these out of model_kwargs into the flat namespace.
    model_kwargs = data.get("model_kwargs") or {}
    for key in ("num_frame_per_block", "timestep_shift"):
        if key in model_kwargs:
            data.setdefault(key, model_kwargs[key])
    if "model_name" in model_kwargs:
        data.setdefault("model_name", model_kwargs["model_name"])
    return config


def install_official_stubs(monkeypatch, log: dict[str, Any]) -> dict[str, Any]:
    """Put fake NVlabs modules on sys.modules so import_official_api resolves."""
    torch_mod = FakeTorch(log)

    omegaconf_mod = types.ModuleType("omegaconf")

    class _OmegaConf:
        @staticmethod
        def load(path):
            return Cfg(_parse_yaml(Path(path).read_text(encoding="utf-8")))

    omegaconf_mod.OmegaConf = _OmegaConf

    pipeline_mod = types.ModuleType("pipeline")

    def _make_pipeline(config, device=None):
        return FakeCausalDiffusionInferencePipeline(config, device=device, log=log)

    pipeline_mod.CausalDiffusionInferencePipeline = _make_pipeline

    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = []

    config_mod = types.ModuleType("utils.config")
    config_mod.normalize_config = fake_normalize_config

    def _setup_nvfp4_pipeline(pipe, config, device, verbose=False):
        if not bool(config.get("model_quant", False)):
            raise ValueError("setup_nvfp4_pipeline requires model_quant=true in the config.")
        log.setdefault("setup", []).append({"pipeline_id": id(pipe), "device": device})
        return pipe

    def _prepare_single_prompt_inputs(config, prompt, device, dtype="bfloat16", batch_size=1, generator=None):
        num_frames = int(config.get("num_output_frames", config.image_or_video_shape[1]))
        per_block = int(config.get("num_frame_per_block", 1))
        if num_frames % per_block != 0:
            raise ValueError(f"num_frames={num_frames} must be divisible by {per_block}")
        latent = list(config.image_or_video_shape)[2:]
        blocks = num_frames // per_block
        noise = torch_mod.randn([batch_size, num_frames, *latent], device=device, dtype=dtype)
        return noise, [[prompt] * blocks for _ in range(batch_size)]

    def _save_video(video, output_path, fps=24):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 64 + b"ftypisom")
        log.setdefault("saved", []).append({"path": str(path), "fps": fps, "shape": video.shape})

    inference_utils_mod = types.ModuleType("utils.inference_utils")
    inference_utils_mod.setup_nvfp4_pipeline = _setup_nvfp4_pipeline
    inference_utils_mod.prepare_single_prompt_inputs = _prepare_single_prompt_inputs
    inference_utils_mod.save_video = _save_video

    dataset_mod = types.ModuleType("utils.dataset")

    def _dataset(data_path, image_size, num_blocks, scene_cut_prefix=""):
        return FakeImagePromptDataset(
            data_path, image_size, num_blocks, scene_cut_prefix=scene_cut_prefix, log=log
        )

    dataset_mod.ImagePromptDataset = _dataset

    misc_mod = types.ModuleType("utils.misc")
    misc_mod.set_seed = lambda seed: log.setdefault("seeds", []).append(int(seed))

    memory_mod = types.ModuleType("utils.memory")
    memory_mod.get_cuda_free_memory_gb = lambda _device: 31.0

    class _DynamicSwapInstaller:
        @staticmethod
        def install_model(model, device=None):
            log.setdefault("dynamic_swap", []).append(device)

    memory_mod.DynamicSwapInstaller = _DynamicSwapInstaller

    utils_mod.config = config_mod
    utils_mod.inference_utils = inference_utils_mod
    utils_mod.dataset = dataset_mod
    utils_mod.misc = misc_mod
    utils_mod.memory = memory_mod

    for name, module in {
        "torch": torch_mod,
        "omegaconf": omegaconf_mod,
        "pipeline": pipeline_mod,
        "utils": utils_mod,
        "utils.config": config_mod,
        "utils.inference_utils": inference_utils_mod,
        "utils.dataset": dataset_mod,
        "utils.misc": misc_mod,
        "utils.memory": memory_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return log
