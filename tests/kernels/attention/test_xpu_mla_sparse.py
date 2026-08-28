# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
)


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]
    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla(device_str, dtype):
    device = torch.device(device_str)
    s_q = 1
    s_kv = 256
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla_masked_chunks(device_str, dtype):
    """Rows whose leading BLOCK_N index entries are all masked must not NaN.

    Regression test: with an -inf running max, a fully-masked leading chunk
    produced re_scale = exp2(-inf - -inf) = NaN, permanently poisoning the
    accumulator even though valid keys followed in later chunks.
    """
    device = torch.device(device_str)
    s_q = 3
    s_kv = 256
    h_q = 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128  # 8 chunks of BLOCK_N=16

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    # row 0: valid keys only in chunks 1-2 -> leading AND trailing masked chunks
    indices[0, 0, 16:48] = torch.arange(32, dtype=torch.int32, device=device)
    # row 1: fully valid
    indices[1, 0, :] = torch.arange(topk, dtype=torch.int32, device=device)
    # row 2: no valid key at all

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.isfinite().all()

    ref_out, _, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out[:2], ref_out[:2], atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits[:2], ref_max_logits[:2], atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse[:2], ref_lse[:2], atol=1e-3, rtol=1e-3)
    # A row with no valid key yields zeros (the reference's convention); its
    # lse/max_logits are large-negative finite rather than the reference's
    # +inf/-inf placeholders, so only the output is compared here.
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_xpu_dense_mha_short_prefill_is_causal():
    class KVProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, 2 * (512 + 3), bias=False)

        def forward(self, value: torch.Tensor):
            return (self.projection(value),)

    device = torch.device("xpu")
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    projection = KVProjection().to(device=device, dtype=dtype)
    impl = XPUMLASparseImpl(
        num_heads=2,
        head_size=576,
        scale=576**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        kv_lora_rank=4,
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        v_head_dim=3,
        kv_b_proj=projection,
    )
    metadata = XPUMLASparseMetadata(
        num_reqs=2,
        max_query_len=3,
        max_seq_len=3,
        num_actual_tokens=5,
        query_start_loc=torch.tensor([0, 2, 5], device=device, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        slot_mapping=torch.empty(5, device=device, dtype=torch.long),
        block_table=torch.empty((2, 1), device=device, dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 0, 1, 1, 1], device=device, dtype=torch.int32),
        use_dense_mha_prefill=True,
    )
    q = torch.randn((5, 2, 576), device=device, dtype=dtype)
    kv_c = torch.randn((5, 4), device=device, dtype=dtype)
    k_pe = torch.randn((5, 64), device=device, dtype=dtype)

    output = impl.forward_mha(q, kv_c, k_pe, metadata)
    projected = projection(kv_c)[0].view(5, 2, 515)
    key = torch.cat((projected[..., :512], k_pe.unsqueeze(1).expand(-1, 2, -1)), dim=-1)
    value = projected[..., 512:]
    expected = torch.empty_like(output)
    for start, end in ((0, 2), (2, 5)):
        scores = torch.einsum("qhd,khd->hqk", q[start:end].float(), key[start:end].float())
        scores.mul_(576**-0.5)
        scores.masked_fill_(~torch.ones(end - start, end - start, device=device, dtype=torch.bool).tril(), float("-inf"))
        expected[start:end] = torch.einsum(
            "hqk,khd->qhd", torch.softmax(scores, dim=-1), value[start:end].float()
        ).to(dtype)

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_xpu_fp8_ds_mla_routes_to_deepklox(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("deepklox")
    device = torch.device("xpu")
    num_tokens, num_heads, topk, block_size = 2, 8, 128, 64
    impl = XPUMLASparseImpl(
        num_heads=num_heads,
        head_size=576,
        scale=576**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_ds_mla",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        kv_lora_rank=512,
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        v_head_dim=256,
        kv_b_proj=torch.nn.Identity(),
        topk_indices_buffer=torch.zeros(
            (num_tokens, topk), device=device, dtype=torch.int32
        ),
    )
    metadata = XPUMLASparseMetadata(
        num_reqs=num_tokens,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        slot_mapping=torch.empty(num_tokens, device=device, dtype=torch.long),
        block_table=torch.tensor([[0], [1]], device=device, dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 1], device=device, dtype=torch.int32),
        block_size=block_size,
        topk_tokens=topk,
        num_decodes=num_tokens,
        num_decode_tokens=num_tokens,
    )
    # A zero-valued, physically contiguous fp8_ds_mla cache is sufficient to
    # verify the vLLM-to-DeepKLOX layout and routing contract.
    kv_cache = torch.zeros((2, block_size, 656), device=device, dtype=torch.uint8)
    q = (
        torch.randn((num_tokens, num_heads, 512), device=device, dtype=torch.bfloat16),
        torch.randn((num_tokens, num_heads, 64), device=device, dtype=torch.bfloat16),
    )

    output, lse = impl.forward_mqa(q, kv_cache, metadata, layer=None)  # type: ignore[arg-type]

    assert lse is None
    assert output.shape == (num_tokens, num_heads, 512)
    torch.testing.assert_close(output, torch.zeros_like(output))
