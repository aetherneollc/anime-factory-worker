"""Kolors ChatGLM encode must survive transformers 5 dropping config.use_cache."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gpu_worker.kolors_compat import patch_kolors_chatglm_transformers5

_UPSTREAM_CONFIG = """\
from transformers import PretrainedConfig


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"

    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        prefix_projection=False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)
"""

_UPSTREAM_MODELING = """\
def forward(self, use_cache=None, return_dict=None):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    return use_cache, return_dict


class Other:
    def forward(self, use_cache=None):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return use_cache
"""

_ENCODER_CONFIG = {
    "num_layers": 2,
    "padded_vocab_size": 32,
    "hidden_size": 8,
    "prefix_projection": False,
    "original_rope": True,
    "use_cache": True,
    "torch_dtype": "float16",
    "eos_token_id": 2,
    "pad_token_id": 0,
}


def _node(root: Path) -> Path:
    node = root / "custom_nodes" / "ComfyUI-Kolors-MZ" / "chatglm3"
    node.mkdir(parents=True)
    (node / "configuration_chatglm.py").write_text(_UPSTREAM_CONFIG, encoding="utf-8")
    (node / "modeling_chatglm.py").write_text(_UPSTREAM_MODELING, encoding="utf-8")
    return node


def test_patch_restores_use_cache_and_is_idempotent(tmp_path: Path):
    node = _node(tmp_path)
    hits = patch_kolors_chatglm_transformers5(tmp_path)
    assert hits == [str(node / "modeling_chatglm.py"), str(node / "configuration_chatglm.py")]
    modeling = (node / "modeling_chatglm.py").read_text(encoding="utf-8")
    config = (node / "configuration_chatglm.py").read_text(encoding="utf-8")
    assert "self.config.use_cache" not in modeling
    assert modeling.count('getattr(self.config, "use_cache", True)') == 2
    assert "self.use_cache = _af_use_cache" in config
    assert 'kwargs.pop("use_cache", True)' in config
    assert patch_kolors_chatglm_transformers5(tmp_path) == []


def test_missing_kolors_node_is_a_noop(tmp_path: Path):
    assert patch_kolors_chatglm_transformers5(tmp_path) == []


def test_patched_config_keeps_use_cache_on_transformers5(tmp_path: Path):
    """transformers 5 pops use_cache; the patched ChatGLMConfig must put it back."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers is not installed")
    major = int(transformers.__version__.split(".", 1)[0])
    if major < 5:
        pytest.skip(f"need transformers>=5, have {transformers.__version__}")

    node = _node(tmp_path)
    patch_kolors_chatglm_transformers5(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "configuration_chatglm_patched",
        node / "configuration_chatglm.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.ChatGLMConfig(**json.loads(json.dumps(_ENCODER_CONFIG)))
    assert cfg.use_cache is True
    assert cfg.original_rope is True
    assert cfg.num_layers == 2
