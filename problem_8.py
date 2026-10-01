# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional

from problem_5 import _flash_attention_forward_gqa_kernel

class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

        _flash_attention_forward_gqa_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # [OPTIONAL BONUS] STUDENT IMPLEMENTATION REQUIRED
        # Implement the Triton backward kernel for GQA from scratch.
        # You should:
        #   1. Precompute delta = sum(dO * O)
        delta = (do * o).sum(dim=-1)

        # expand kv heads to match q heads
        n_grp = n_heads // n_kv_heads
        k_exp = k.repeat_interleave(n_grp, dim=1)
        v_exp = v.repeat_interleave(n_grp, dim=1)

        #   2. Recompute attention probabilities P = softmax(QK^T)
        s = q @ k_exp.transpose(-1, -2) 
        s *= ctx.softmax_scale
        causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=s.device, dtype=s.dtype).triu(diagonal=1)
        s += causal_mask
        P = torch.softmax(s , dim=-1)

        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        #   4. Respect GQA mapping and causal mask
        dv = P.transpose(-1, -2) @ do
        dP = do @ v_exp.transpose(-1, -2)
        ds = P * (dP - delta.unsqueeze(-1))

        dq = ds @ k_exp * ctx.softmax_scale
        dk = ds.transpose(-1, -2) @ q * ctx.softmax_scale

        # compress back to original kv heads
        dk = dk.reshape(batch, n_kv_heads, n_grp, seq_len, head_dim).sum(dim=2)
        dv = dv.reshape(batch, n_kv_heads, n_grp, seq_len, head_dim).sum(dim=2)
        
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)