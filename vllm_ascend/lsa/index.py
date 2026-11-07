import logging

import torch
import triton
import triton.language as tl

from vllm_ascend.lsa.read_paged import read_pool_rows

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Decode-side LSA index over the paged KV pool, one instance per full-attention layer.
# Keys live in a slot-indexed buffer (unit rows, normalize(k_pf @ W_K)); when a request's
# cluster block closes, its keys are clustered on device (lsa.core ward) and the buckets are
# appended to per-layer centroid/member row pools at a per-request base row. Decode selection
# scores the request's buckets (offset scan), picks top-p, expands members (positions) to a
# read list, maps positions to pool slots through the step's slot table, and reads.


@triton.jit
def _scan_offset_kernel(
    q_ptr,
    c_ptr,
    base_ptr,
    counts_ptr,
    out_ptr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    program = tl.program_id(0)
    request = tl.program_id(1)
    rows = program * ROWS + tl.arange(0, ROWS)
    inside = rows < tl.num_programs(0) * ROWS
    visible = rows < tl.load(counts_ptr + request)
    base = tl.load(base_ptr + request).to(tl.int64)
    acc = tl.zeros([ROWS], dtype=tl.float32)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + request * D + dims).to(tl.float32)
    c = tl.load(
        c_ptr + (base + rows[:, None]) * D + dims[None, :],
        mask=visible[:, None],
        other=0.0,
    )
    acc = tl.sum(c.to(tl.float32) * q[None, :], axis=1)
    out = tl.where(visible, acc, float("-inf"))
    tl.store(out_ptr + request.to(tl.int64) * tl.num_programs(0) * ROWS + rows, out, mask=inside)


def scan_offset(index_queries, centroids, base, counts, max_rows):
    # index_queries [B, D] bf16; centroids [pool_rows, D] bf16; base [B] int64; counts [B] int32
    batch, d = index_queries.shape
    rows = triton.cdiv(max_rows, 32) * 32
    out = torch.empty(batch, rows, device=index_queries.device, dtype=torch.float32)
    _scan_offset_kernel[(rows // 32, batch)](
        index_queries, centroids, base, counts, out, D=d, ROWS=32
    )
    return out


class LsaConfig:
    def __init__(self, block):
        block = block or {}
        indexer = block.get("indexer", {})
        serving = block.get("serving", {})
        self.window = int(indexer.get("window_size", 512))
        self.sink = int(indexer.get("sink_size", 64))
        self.top_k = int(indexer.get("top_k", 1024))
        self.top_p = int(indexer.get("top_p", 64))
        self.pool_size = int(indexer.get("pool_size", 16))
        self.bucket_size = int(indexer.get("bucket_size", 64))
        self.bucket_cap = int(indexer.get("bucket_cap", 128))
        self.cluster_block = int(indexer.get("cluster_block_size", 1024))
        self.projections = serving.get("projections", "")
        self.min_context = int(serving.get("min_context", 2048))
        for name in ("window", "sink", "top_k", "top_p", "pool_size", "bucket_size", "bucket_cap", "cluster_block"):
            if getattr(self, name) <= 0:
                raise ValueError(f"lsa config {name} must be positive")


def make_index_key(k: torch.Tensor, wk: torch.Tensor, rotary_dim: int, kv_heads: int) -> torch.Tensor:
    # k [T, Hkv, head_dim] post-rope; position-free dims projected and unit-normalized
    k_pf = k[:, :, rotary_dim:].reshape(k.shape[0], -1).float()
    keys = k_pf @ wk
    return torch.nn.functional.normalize(keys, dim=-1).to(torch.bfloat16)


def make_index_query(q: torch.Tensor, wq: torch.Tensor, rotary_dim: int, kv_heads: int) -> torch.Tensor:
    # q [T, Hq, head_dim] post-rope; per-group fp32 sum of position-free dims, projected, normalized
    group = q.shape[1] // kv_heads
    q_pf = q[:, :, rotary_dim:].reshape(q.shape[0], kv_heads, group, -1)
    grouped = q_pf.sum(2, dtype=torch.float32).reshape(q.shape[0], -1)
    queries = grouped @ wq
    return torch.nn.functional.normalize(queries, dim=-1).to(torch.bfloat16)


class ReqIndex:
    def __init__(self):
        self.base_row = -1
        self.closed_blocks = 0


class LayerIndex:
    def __init__(self, layer_id: int, wq: torch.Tensor, wk: torch.Tensor, config: LsaConfig, device):
        self.layer_id = layer_id
        self.wq = wq
        self.wk = wk
        self.config = config
        self.device = device
        self.requests: dict[int, ReqIndex] = {}
        self.row_cursor = 0
        self.index_keys = None
        self.centroids = None
        self.members = None
        self.block_bucket_count = config.cluster_block // config.bucket_size
        self.max_rows = 0

    def ensure_pools(self, pool_rows: int):
        if self.index_keys is not None:
            return
        self.max_rows = pool_rows
        d = self.wk.shape[1]
        # Slot-indexed key rows cover the whole KV pool; bucket rows are
        # cluster_block/bucket_size per closed block, a small fraction.
        self.index_keys = torch.zeros(pool_rows, d, device=self.device, dtype=torch.bfloat16)
        bucket_rows = max(pool_rows // self.config.cluster_block, 1024) * self.block_bucket_count
        self.bucket_rows = bucket_rows
        self.centroids = torch.zeros(bucket_rows, d, device=self.device, dtype=torch.bfloat16)
        self.members = torch.zeros(bucket_rows, self.config.bucket_cap, device=self.device, dtype=torch.int32)
        logger.info("LSA layer %d pools slots=%d bucket_rows=%d dims=%d", self.layer_id, pool_rows, bucket_rows, d)

    def append_keys(self, slots: torch.Tensor, keys: torch.Tensor):
        mask = slots >= 0
        if bool(mask.all()):
            self.index_keys[slots.long()] = keys
        else:
            self.index_keys[slots[mask].long()] = keys[mask]

    def ensure_blocks(self, req_id: int, seq_len: int, gather_block):
        # gather_block(block_index) -> index keys [cluster_block, D] bf16 for one closed block
        state = self.requests.setdefault(req_id, ReqIndex())
        closed = seq_len // self.config.cluster_block
        for block_index in range(state.closed_blocks, closed):
            keys = gather_block(block_index)
            buckets = build_buckets(keys, self.config)
            count = buckets.bucket_centroids.shape[0]
            if state.base_row < 0:
                state.base_row = self.row_cursor
            start = self.row_cursor
            if start + count > self.bucket_rows:
                raise RuntimeError(f"LSA layer {self.layer_id} row pool exhausted")
            self.centroids[start : start + count] = buckets.bucket_centroids
            self.members[start : start + count] = buckets.bucket_members
            self.row_cursor += count
            state.closed_blocks = block_index + 1

    def visible_counts(self, req_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.tensor(
            [self.requests[r].closed_blocks * self.block_bucket_count if r in self.requests else 0 for r in req_ids],
            device=self.device, dtype=torch.int32,
        )
        base = torch.tensor(
            [max(self.requests[r].base_row, 0) for r in req_ids],
            device=self.device, dtype=torch.int64,
        )
        return counts, base

    def select_decode(
        self,
        q_idx: torch.Tensor,
        req_ids: list[int],
        positions: torch.Tensor,
        slot_table: torch.Tensor,
    ) -> torch.Tensor:
        from lsa.core.kernels.triton.keys import unpack_ids
        from lsa.core.kernels.triton.top_k import top_keys
        from lsa.core.kernels.triton.expansion import expand_in_order

        counts, base = self.visible_counts(req_ids)
        max_rows = int(counts.max().item()) if counts.numel() else 0
        batch = q_idx.shape[0]
        if max_rows == 0:
            return None
        scores = scan_offset(q_idx, self.centroids, base, counts, max_rows)
        packed = top_keys(scores, counts, self.config.top_p)
        picked = unpack_ids(packed)
        global_rows = torch.where(picked >= 0, picked + base.to(torch.int32)[:, None], picked)
        lists, _ = expand_in_order(
            global_rows,
            self.members,
            positions.to(torch.int32),
            self.config.window,
            self.config.sink,
            self.config.top_k,
            slot_table=slot_table,
        )
        return lists


def build_buckets(keys: torch.Tensor, config: LsaConfig):
    from lsa.core.reference import build_buckets as reference_build_buckets

    return reference_build_buckets(
        keys, config.pool_size, config.bucket_size, config.bucket_cap, config.cluster_block
    )


class LsaState:
    # One per worker process; holds per-layer indexes and the step cache
    def __init__(self, config: LsaConfig, layer_ids: list[int], device):
        from vllm_ascend.lsa.weights import load_projections

        self.config = config
        self.projections = load_projections(config.projections, layer_ids, device)
        self.layers = {
            layer_id: LayerIndex(layer_id, *self.projections[layer_id], config, device)
            for layer_id in layer_ids
        }
        self.step_cache = {}
        self.engaged = False
        self.shape_logged = False
        logger.info("LSA armed layers=%s min_context=%d", layer_ids, config.min_context)


_STATE = None


def get_state() -> LsaState:
    return _STATE


def set_state(state: LsaState):
    global _STATE
    _STATE = state
