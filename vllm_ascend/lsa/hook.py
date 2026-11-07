import logging
import os
import re

import torch

from vllm.forward_context import get_forward_context

from vllm_ascend.lsa.index import LsaConfig, LsaState, get_state, make_index_key, make_index_query, set_state

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _try_arm(attn_module):
    config = attn_module.config
    block = getattr(config, "lsa", None)
    if block is None:
        hf = getattr(config, "hf_config", None)
        block = getattr(hf, "lsa", None) if hf is not None else None
    if block is None:
        return False
    layer_types = getattr(config, "layer_types", None) or []
    layer_ids = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    if not layer_ids:
        return False
    cfg = LsaConfig(block if isinstance(block, dict) else dict(block))
    device = next(attn_module.parameters()).device
    set_state(LsaState(cfg, layer_ids, device))
    return True


def _step_cache(md, key, builder):
    state = get_state()
    cache_key = (id(md), key)
    if cache_key not in state.step_cache:
        if len(state.step_cache) > 64:
            state.step_cache.clear()
        state.step_cache[cache_key] = builder()
    return state.step_cache[cache_key]


def lsa_forward_or_fallback(attn_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    try:
        return _lsa_forward(attn_module, q, k, v)
    except Exception:
        logger.exception("LSA forward failed, falling back to dense")
        return attn_module(q, k, v)


def _lsa_forward(attn_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if os.environ.get("VLLM_LSA_DISABLE") == "1":
        return attn_module(q, k, v)
    state = get_state()
    if state is None and not _try_arm(attn_module):
        return attn_module(q, k, v)
    state = get_state()

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return attn_module(q, k, v)
    metadata = attn_metadata.get(attn_module.layer_name)
    if metadata is None or metadata.attn_state.name != "DecodeOnly":
        return attn_module(q, k, v)
    if getattr(forward_context, "cudagraph_runtime_mode", None) is not None and str(
        forward_context.cudagraph_runtime_mode
    ) not in ("CUDAGraphMode.NONE", "NONE"):
        return attn_module(q, k, v)

    match = _LAYER_RE.search(attn_module.layer_name or "")
    layer_id = int(match.group(1)) if match else -1
    layer_index = state.layers.get(layer_id)
    if layer_index is None:
        return attn_module(q, k, v)

    cfg = state.config
    kv_cache = attn_module.kv_cache
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    block_size = key_cache.shape[1]
    num_tokens, q_heads = q.shape[0], q.shape[1]
    batch = metadata.num_actual_tokens
    if batch == 0 or metadata.seq_lens_list is None:
        return attn_module(q, k, v)
    seq_lens = list(metadata.seq_lens_list)[:batch]
    if min(seq_lens) < max(cfg.min_context, cfg.cluster_block + cfg.window):
        return attn_module(q, k, v)

    block_tables = metadata.block_tables[:batch]
    slots = metadata.slot_mapping[:batch]

    # 1. write the new k/v rows into the pool ourselves (self.attn is bypassed)
    k_pool = key_cache.reshape(-1, kv_heads, head_dim)
    v_pool = value_cache.reshape(-1, kv_heads, head_dim)
    rotary_dim = int(getattr(attn_module.rotary_emb, "rotary_dim", 0)) or head_dim // 4
    k_rows = k[:batch].view(batch, kv_heads, head_dim)
    v_rows = v[:batch].view(batch, kv_heads, head_dim)
    k_pool[slots.long()] = k_rows
    v_pool[slots.long()] = v_rows

    # 2. index keys of the new tokens, slot-indexed
    layer_index.ensure_pools(k_pool.shape[0])
    wk = layer_index.wk
    keys = make_index_key(k_rows, wk, rotary_dim, kv_heads)
    layer_index.append_keys(slots, keys)

    # 3. per-request host state and closed-block clustering
    req_keys = _step_cache(metadata, "req_keys", lambda: _req_keys_of(block_tables, block_size, batch))
    for row, seq_len in enumerate(seq_lens):
        req_id = req_keys[row]
        block_table_row = block_tables[row]

        def gather_block(block_index, row=row, block_table_row=block_table_row):
            pos = torch.arange(
                block_index * cfg.cluster_block, (block_index + 1) * cfg.cluster_block,
                device=block_table_row.device, dtype=torch.int64,
            )
            slot_ids = block_table_row[pos // block_size].long() * block_size + (pos % block_size)
            return layer_index.index_keys[slot_ids]

        layer_index.ensure_blocks(req_id, seq_len, gather_block)

    # 4. position -> slot table for the expansion, one per step (shared across layers)
    slot_table = _step_cache(metadata, "slot_table", lambda: _slot_table_of(block_tables, block_size, max(seq_lens)))

    # 5. select and read
    q_rows = q[:batch]
    wq = layer_index.wq
    q_idx = make_index_query(q_rows, wq, rotary_dim, kv_heads)
    positions = torch.tensor([s - 1 for s in seq_lens], device=q.device, dtype=torch.int32)
    scale = attn_module.scale
    out_rows = layer_index.select_decode(
        q_idx, q_rows, req_keys, positions, slot_table, k_pool, v_pool, scale
    )
    if out_rows is None:
        return attn_module(q, k, v)
    if out_rows.dtype != q.dtype:
        out_rows = out_rows.to(q.dtype)
    out = torch.zeros(num_tokens, q_heads * head_dim, device=q.device, dtype=q.dtype)
    out[:batch] = out_rows.reshape(batch, q_heads * head_dim)
    return out


def _req_keys_of(block_tables, block_size, batch):
    return [int(x) for x in block_tables[:batch, 0].cpu().tolist()]


def _slot_table_of(block_tables, block_size, max_len):
    ctx = ((max_len + block_size - 1) // block_size) * block_size
    pos = torch.arange(ctx, device=block_tables.device, dtype=torch.int64)
    bi, off = pos // block_size, pos % block_size
    table = block_tables[:, bi].long() * block_size + off
    return table.to(torch.int32)
