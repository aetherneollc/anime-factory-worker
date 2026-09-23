"""Keep ComfyUI-Kolors-MZ ChatGLM encode working on transformers 5.

transformers>=5 ``PreTrainedConfig`` drops generation kwargs. ``use_cache`` is
popped and never stored, so ``MZ_ChatGLM3_Advance_V2`` raises
``AttributeError: 'ChatGLMConfig' object has no attribute 'use_cache'``.
"""

from __future__ import annotations

import os
from pathlib import Path

_USE_CACHE_OLD = "use_cache = use_cache if use_cache is not None else self.config.use_cache"
_USE_CACHE_NEW = (
    'use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)'
)
_CONFIG_OLD = (
    "        self.prefix_projection = prefix_projection\n"
    "        super().__init__(**kwargs)\n"
)
_CONFIG_NEW = (
    "        self.prefix_projection = prefix_projection\n"
    "        # transformers>=5 pops generation kwargs and does not set use_cache.\n"
    "        _af_use_cache = kwargs.pop(\"use_cache\", True)\n"
    "        _af_original_rope = kwargs.pop(\"original_rope\", True)\n"
    "        super().__init__(**kwargs)\n"
    "        self.use_cache = _af_use_cache\n"
    "        if not hasattr(self, \"original_rope\"):\n"
    "            self.original_rope = _af_original_rope\n"
)
_CONFIG_MARKER = "self.use_cache = _af_use_cache"


def patch_kolors_chatglm_transformers5(comfy_dir: Path | str | None = None) -> list[str]:
    """Patch the baked Kolors-MZ ChatGLM sources. Idempotent; missing node is a no-op."""
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    node = root / "custom_nodes" / "ComfyUI-Kolors-MZ" / "chatglm3"
    hits: list[str] = []
    modeling = node / "modeling_chatglm.py"
    config = node / "configuration_chatglm.py"
    if modeling.is_file():
        text = modeling.read_text(encoding="utf-8")
        if _USE_CACHE_OLD in text:
            modeling.write_text(text.replace(_USE_CACHE_OLD, _USE_CACHE_NEW), encoding="utf-8")
            hits.append(str(modeling))
    if config.is_file():
        text = config.read_text(encoding="utf-8")
        if _CONFIG_MARKER not in text and _CONFIG_OLD in text:
            config.write_text(text.replace(_CONFIG_OLD, _CONFIG_NEW, 1), encoding="utf-8")
            hits.append(str(config))
    return hits
