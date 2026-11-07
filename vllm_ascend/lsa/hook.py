import logging
import os
import re

import torch
import torch_npu

from vllm.forward_context import get_forward_context

from vllm_ascend.lsa.index import LsaConfig, LsaState, get_state, make_index_key, make_index_query, set_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _try_arm(attn_module):
    config = attn_module.config
    block = getattr(config, "lsa", None)
    if block is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            root = getattr(vllm_config.model_config, "hf_config", None)
            block = getattr(root, "lsa", None) if root is not None else None
        except Exception:
            block = None
    if block is None:
        logger.info("LSA not armed: no lsa config block (config=%s)", type(config).__name__)
        return False
    layer_types = getattr(config, "layer_types", None) or []
    layer_ids = [i for i, kind in enumerate(layer_types) if "full" in str(kind)]
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
        return attn_module.attn(q, k, v)


def _read_mxfp8(q_rows, lists, key_cache, value_cache, k_scale_cache, v_recip, scale):
    # lists [B, K] absolute slots (-1 padded); PA_NZ cache
    # key/value [num_blocks, kvh, D//32, block_size, 32] fp8,
    # k_scale [num_blocks, kvh, block_size//16, D//64, 16, 2] e8m0 bytes.
    batch = q_rows.shape[0]
    kvh = key_cache.shape[1]
    num_groups = key_cache.shape[2]
    block_size = key_cache.shape[3]
    head_dim = num_groups * 32
    slot = lists.long().clamp_min(0)
    blk = slot // block_size
    off = slot % block_size
    kbytes = key_cache.view(torch.uint8)
    kdata = (
        kbytes[blk, :, :, off, :].contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        .view(batch, -1, kvh, num_groups, 32)
    )
    row = off // 16
    frag = off % 16
    ksc = k_scale_cache[blk, :, row, :, frag, :].to(torch.int32)
    ksc = torch.pow(2.0, (ksc - 127).to(torch.float32)).unsqueeze(-1)
    k_sel = (kdata.view(batch, -1, kvh, num_groups // 2, 2, 32) * ksc).view(batch, -1, kvh, head_dim).to(q_rows.dtype)
    vdata = (
        value_cache.view(torch.uint8)[blk, :, :, off, :].contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        .view(batch, -1, kvh, head_dim)
    )
    v_sel = (vdata / v_recip.to(torch.float32).view(kvh, head_dim)).to(q_rows.dtype)
    live = (lists >= 0).to(q_rows.dtype).unsqueeze(-1).unsqueeze(-1)
    v_sel = v_sel * live
    qg = q_rows.view(batch, kvh, -1, head_dim).to(torch.float32)
    logits = torch.einsum("bhgd,bkhd->bhgk", qg, k_sel.float()) * scale
    logits = logits.masked_fill(lists[:, None, None, :] < 0, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("bhgk,bkhd->bhgd", probs, v_sel.float())
    return out.reshape(batch, -1, head_dim).to(q_rows.dtype)


def _lsa_forward(attn_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if os.environ.get("VLLM_LSA_DISABLE") == "1":
        return attn_module.attn(q, k, v)
    state = get_state()
    if state is None and not _try_arm(attn_module):
        return attn_module.attn(q, k, v)
    state = get_state()

    backend = attn_module.attn
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return backend(q, k, v)
    metadata = attn_metadata.get(getattr(backend, "layer_name", None))
    if metadata is None:
        return backend(q, k, v)

    layer_id = getattr(backend, "layer_idx", None)
    if layer_id is None:
        match = _LAYER_RE.search(getattr(backend, "layer_name", "") or "")
        layer_id = int(match.group(1)) if match else -1
    layer_index = state.layers.get(layer_id)
    if layer_index is None:
        return backend(q, k, v)

    # Index keys must cover prefill tokens too: decode-time bucket building
    # gathers whole cluster blocks whose rows were written at prefill.
    if metadata.attn_state.name != "DecodeOnly":
        _append_step_keys(attn_module, layer_index, metadata, k)
        return backend(q, k, v)
    if getattr(forward_context, "cudagraph_runtime_mode", None) is not None and str(
        forward_context.cudagraph_runtime_mode
    ) not in ("CUDAGraphMode.NONE", "NONE"):
        return backend(q, k, v)

    cfg = state.config
    num_tokens = q.shape[0]
    batch = metadata.num_actual_tokens
    if batch == 0 or metadata.seq_lens_list is None:
        return backend(q, k, v)
    seq_lens = list(metadata.seq_lens_list)[:batch]
    if min(seq_lens) < max(cfg.min_context, cfg.cluster_block + cfg.window):
        return backend(q, k, v)
    if not state.engaged:
        state.engaged = True
        logger.info("LSA engaged: layer=%s batch=%s seq_lens=%s", layer_id, batch, seq_lens[:4])

    kv_heads = attn_module.num_kv_heads
    head_dim = attn_module.head_dim
    kv_cache = getattr(attn_module, "kv_cache", None) or getattr(backend, "kv_cache", None)
    if kv_cache is None or len(kv_cache) < 4:
        logger.warning("LSA: mxfp8 kv_cache tuple expected, got %s", type(kv_cache))
        return backend(q, k, v)
    key_cache, value_cache, k_scale_cache = kv_cache[0], kv_cache[1], kv_cache[2]
    block_size = key_cache.shape[3]
    block_tables = metadata.block_tables[:batch]
    slots = metadata.slot_mapping[:batch]

    # 1. quantize the new k/v rows and write through the backend's PA_NZ scatter
    k_rows = k[:batch].view(batch, kv_heads, head_dim)
    v_rows = v[:batch].view(batch, kv_heads, head_dim)
    key_mxfp8, key_scale = torch_npu.npu_dynamic_mx_quant(k_rows, dst_type=torch.float8_e4m3fn)
    v_recip = backend.v_cache_scale_float_reciprocal
    value_mxfp8 = torch_npu.npu_quantize(
        v_rows.view(batch, -1), v_recip, None, torch.float8_e4m3fn, -1, False
    ).view(batch, kv_heads, head_dim)
    backend.impl.reshape_and_cache(key_mxfp8, value_mxfp8, key_scale, kv_cache, metadata)

    # 2. index keys of the new tokens, slot-indexed
    layer_index.ensure_pools(key_cache.shape[0] * block_size)
    wk = layer_index.wk
    rotary_dim = int(getattr(attn_module.rotary_emb, "rotary_dim", 0)) or head_dim // 4
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
    q_rows = q[:batch].view(batch, -1, head_dim)
    wq = layer_index.wq
    q_idx = make_index_query(q_rows, wq, rotary_dim, kv_heads)
    positions = torch.tensor([s - 1 for s in seq_lens], device=q.device, dtype=torch.int32)
    scale = float(1.0 / head_dim ** 0.5)
    lists = layer_index.select_decode(q_idx, req_keys, positions, slot_table)
    if lists is None:
        return backend(q, k, v)
    out_rows = _read_mxfp8(q_rows, lists, key_cache, value_cache, k_scale_cache, v_recip, scale)
    out = torch.zeros(num_tokens, q.shape[1], device=q.device, dtype=q.dtype)
    out[:batch] = out_rows.reshape(batch, -1)
    return out


def _append_step_keys(attn_module, layer_index, metadata, k: torch.Tensor):
    n = metadata.num_actual_tokens
    if n <= 0:
        return
    kv_cache = getattr(attn_module, "kv_cache", None) or getattr(attn_module.attn, "kv_cache", None)
    if kv_cache is None or len(kv_cache) < 4:
        return
    key_cache = kv_cache[0]
    block_size = key_cache.shape[3]
    layer_index.ensure_pools(key_cache.shape[0] * block_size)
    kv_heads = attn_module.num_kv_heads
    head_dim = attn_module.head_dim
    rotary_dim = int(getattr(attn_module.rotary_emb, "rotary_dim", 0)) or head_dim // 4
    k_rows = k[:n].view(n, kv_heads, head_dim)
    keys = make_index_key(k_rows, layer_index.wk, rotary_dim, kv_heads)
    layer_index.append_keys(metadata.slot_mapping[:n], keys)


def _req_keys_of(block_tables, block_size, batch):
    return [int(x) for x in block_tables[:batch, 0].cpu().tolist()]


def _slot_table_of(block_tables, block_size, max_len):
    ctx = ((max_len + block_size - 1) // block_size) * block_size
    pos = torch.arange(ctx, device=block_tables.device, dtype=torch.int64)
    bi, off = pos // block_size, pos % block_size
    table = block_tables[:, bi].long() * block_size + off
    return table.to(torch.int32)
