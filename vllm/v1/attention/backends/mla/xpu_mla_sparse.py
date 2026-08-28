# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)


class XPUMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "XPU_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["XPUMLASparseMetadata"]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["XPUMLASparseMetadataBuilder"]:
        return XPUMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["XPUMLASparseImpl"]:
        return XPUMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]


@dataclass
class XPUMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    block_size: int = 1
    topk_tokens: int = 2048
    prefill_max_seq_len: int = 0

    # The shared MLA layer (`mla_attention.py::forward_impl`) reads these
    # decode/prefill counts unconditionally for every MLA metadata (it asserts
    # `num_decodes`/`num_prefills`/`num_decode_tokens is not None` and uses
    # `num_decode_tokens` to split MQA vs dense-MHA tokens). The CUDA sparse
    # backends carry them via `SparseMLACommonMetadataBuilder`; this XPU backend
    # builds its own metadata and previously omitted them, so a sparse-MLA
    # (DeepSeek / GLM DSA) run on XPU crashed with
    # `'XPUMLASparseMetadata' object has no attribute 'num_decode_tokens'`.
    # This backend serves both prefill and decode through the top-k sparse MQA
    # path (see `forward_mqa`), so all tokens are routed as "decode"
    # (`num_decode_tokens == num_actual_tokens`, `num_prefills == 0`); that keeps
    # the shared layer's `num_mha_tokens` at 0 and never enters the dense-MHA
    # prefill branch (which needs prefill-only fields this backend lacks).
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    use_dense_mha_prefill: bool = False


@dataclass
class XPUMLASparseMetadataBuilder(AttentionMetadataBuilder[XPUMLASparseMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XPUMLASparseMetadata:
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=1,
            treat_short_extends_as_decodes=True,
            require_uniform=False,
        )
        query_lens = np.diff(
            np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        )
        seq_lens = np.asarray(common_attn_metadata.seq_lens_cpu_upper_bound)
        has_cached_prefix = bool(np.any(seq_lens - query_lens > 0))
        use_dense_mha_prefill = (
            num_prefills > 0
            and num_decode_tokens == 0
            and common_attn_metadata.max_seq_len <= self.topk_tokens
            and not has_cached_prefix
            and not self.vllm_config.attention_config.sparse_mla_force_mqa
        )
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        metadata = XPUMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            prefill_max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            use_dense_mha_prefill=use_dense_mha_prefill,
        )
        return metadata


class XPUMLASparseImpl(MLAAttentionImpl[XPUMLASparseMetadata]):
    is_sparse = True
    supports_dense_mha_prefill = True
    masked_mha_available = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Optional["Indexer"] = None,
        **mla_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.v_head_dim: int = mla_args["v_head_dim"]
        self.kv_b_proj = mla_args["kv_b_proj"]
        self.softmax_scale = scale
        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache_dtype == "fp8_ds_mla":
            return
        super().do_kv_cache_update(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        if not attn_metadata.use_dense_mha_prefill:
            raise RuntimeError("XPU dense MHA was selected without eligible metadata")

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        key = torch.cat(
            [k_nope, k_pe.unsqueeze(1).expand(-1, self.num_heads, -1)], dim=-1
        )
        output = torch.empty(
            (q.shape[0], self.num_heads, self.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        starts = attn_metadata.query_start_loc_cpu.tolist()
        for start, end in zip(starts[:-1], starts[1:]):
            output[start:end] = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                key[start:end].transpose(0, 1).unsqueeze(0),
                value[start:end].transpose(0, 1).unsqueeze(0),
                is_causal=True,
                scale=self.softmax_scale,
            ).squeeze(0).transpose(0, 1)
        return output

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        topk_indices = topk_indices.view(num_tokens, 1, -1)

        output, _, _ = triton_bf16_mla_sparse_interface(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
        )

        return output[:, : self.num_heads, :]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if self.kv_cache_dtype == "fp8_ds_mla":
            try:
                from deepklox import flash_mla_with_kvcache
            except ImportError as error:
                raise RuntimeError(
                    "XPU fp8_ds_mla requires DeepKLOX. Add "
                    "/workspace/applications.ai.gpu.deepklox to PYTHONPATH."
                ) from error

            _, block_stride_rows = flat_kv_row_view(
                kv_c_and_k_pe_cache, attn_metadata.block_size
            )
            physical_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
            )
            output, _ = flash_mla_with_kvcache(
                q=q.unsqueeze(1),
                k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2).view(
                    torch.float8_e4m3fn
                ),
                block_table=None,
                cache_seqlens=None,
                head_dim_v=self.kv_lora_rank,
                is_fp8_kvcache=True,
                indices=physical_indices.unsqueeze(1),
                softmax_scale=self.softmax_scale,
            )
            return output.squeeze(1), None

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                f"Unsupported XPU sparse MLA KV cache dtype: {self.kv_cache_dtype}"
            )

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        attn_out = self._forward_bf16_kv(q, kv_rows, topk_indices_global, attn_metadata)

        return attn_out, None
