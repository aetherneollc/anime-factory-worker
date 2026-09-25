"""Replace HunyuanImage-2.1 flash-attn import with an SDPA fallback.

5090 / sm_120 has no usable flash-attn 2.7.3 wheel. The official module imports
flash_attn at import time and cannot load without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("/opt/HunyuanImage-2.1/hyimage/models/hunyuan/modules/flash_attn_no_pad.py")

PATCH = r'''import torch
from einops import rearrange

# SDPA_FALLBACK — used when flash_attn is unavailable (e.g. RTX 5090 / sm_120).
_HAS_FLASH = False
try:
    from flash_attn_interface import flash_attn_varlen_func  # noqa: F401
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    _HAS_FLASH = True
    print("Using FlashAttention v3.")
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn import flash_attn_varlen_qkvpacked_func
        from flash_attn.bert_padding import pad_input, unpad_input
        _HAS_FLASH = True
        print("FlashAttention v3 not found, falling back to v2.")
    except ImportError:
        print("FlashAttention not found, using PyTorch SDPA fallback.")


def get_cu_seqlens(text_mask: torch.Tensor, img_len: int):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=text_mask.device)
    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2
    return cu_seqlens, max_len


def flash_attn_v3(q, k, v, cu_seqlens, max_s, causal=False, deterministic=False):
    if not _HAS_FLASH:
        qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=causal)
        return out.transpose(1, 2)
    batch_size, seqlen = q.shape[:2]
    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    output = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, causal=causal, deterministic=deterministic
    )
    return output.view(batch_size, seqlen, *output.shape[-2:])


def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    if not _HAS_FLASH:
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        return out.transpose(1, 2)
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)[:4]
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_seqlens, max_s, dropout_p,
        softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
    )
    if isinstance(output_unpad, tuple):
        output_unpad = output_unpad[0]
    output = pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen)
    return rearrange(output, "b s (h d) -> b s h d", h=nheads)
'''


def main() -> int:
    if not TARGET.is_file():
        print(f"missing {TARGET}", file=sys.stderr)
        return 1
    TARGET.write_text(PATCH, encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
