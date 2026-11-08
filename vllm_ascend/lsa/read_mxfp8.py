import logging

import torch

logger = logging.getLogger(__name__)


def _bitcast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # NPU view() between float8/uint8 value-casts on this stack; build the
    # reinterpret view over the same untyped storage instead.
    shape = list(t.shape)
    out = torch.empty(0, dtype=dtype, device=t.device)
    return out.set_(t.untyped_storage(), 0, shape, None)


def _block_local(blk, num_blocks):
    # map global block ids to positions inside the deduplicated block list
    blk_u = torch.unique(blk)
    loc = torch.full((num_blocks,), -1, dtype=torch.long, device=blk.device)
    loc[blk_u] = torch.arange(blk_u.shape[0], device=blk.device)
    return blk_u, loc[blk]


def _decode_e4m3(b):
    # b: signed byte values as float32 (exact). Recover the unsigned byte
    # first (sign bit 7), then decode e4m3fn magnitude bits.
    uu = torch.where(b < 0, b + 256.0, b)
    sgn = torch.where(uu >= 128, -1.0, 1.0)
    u = uu - torch.floor(uu / 128) * 128
    e = torch.floor(u / 8)
    m = u - e * 8
    normal = (1.0 + m / 8) * torch.exp2(e - 7)
    subnormal = (m / 8) * 2.0 ** -6
    return torch.where(e <= 0, subnormal, normal) * sgn


def gather_dequant_mxfp8(lists, key_cache, value_cache, k_scale_cache, v_recip):
    # lists [B, K] int32 absolute slots (-1 padded); PA_NZ mxfp8 caches:
    # key/value [num_blocks, kvh, D//32, block_size, 32] fp8,
    # k_scale [num_blocks, kvh, block_size//16, D//64, 16, 2] e8m0 bytes.
    # Returns K [N, kvh, D] and V [N, kvh, D] fp32, N = B*K.
    # NPU-op constraints found on 950DT/CANN 9.2.1: advanced indexing and
    # torch.gather corrupt or no-op on uint8/fp8/int8, index_select works on
    # int8 only for dim0 of the original layout, torch.gather/take only on
    # fp32, and triton-ascend drops data-dependent load addresses. So:
    # dedup blocks, 4D int8 index_select, cast to fp32, then flat torch.take
    # with computed int64 offsets and arithmetic e4m0/e8m0 decode.
    bsz, k = lists.shape
    kvh = key_cache.shape[1]
    groups = key_cache.shape[2]
    blk_size = key_cache.shape[3]
    sub = key_cache.shape[4]
    d = groups * sub
    nb = key_cache.shape[0]
    n = bsz * k
    slot = lists.reshape(-1).long().clamp_min(0)
    live = (lists.reshape(-1) >= 0).to(torch.float32)
    blk = (slot // blk_size).clamp(0, nb - 1)
    off = slot % blk_size
    blk_u, local = _block_local(blk, nb)
    row = off // 16
    frag = off % 16

    h = torch.arange(kvh, device=lists.device)
    g = torch.arange(groups, device=lists.device)
    s = torch.arange(sub, device=lists.device)

    kflat = _bitcast(key_cache, torch.int8).index_select(0, blk_u).to(torch.float32).reshape(-1)
    k_off = (local * (kvh * groups * blk_size * sub))[:, None, None, None] \
        + (off * sub)[:, None, None, None] \
        + (h * (groups * blk_size * sub))[None, :, None, None] \
        + (g * blk_size * sub)[None, None, :, None] + s[None, None, None, :]
    kbyte = torch.gather(kflat, 0, k_off.reshape(-1))
    kval = _decode_e4m3(kbyte).view(n, kvh, groups, sub)

    sflat = _bitcast(k_scale_cache, torch.int8).index_select(0, blk_u).to(torch.float32).reshape(-1)
    sshape = k_scale_cache.shape
    s_strides = [int(k_scale_cache.stride(i)) for i in range(6)]
    g64 = torch.arange(d // 64, device=lists.device)
    sb = torch.arange(2, device=lists.device)
    s_off = (local * s_strides[0])[:, None, None, None, None, None] \
        + (h * s_strides[1])[None, :, None, None, None, None] \
        + (row * s_strides[2])[:, None, None, None, None, None] \
        + (g64 * s_strides[3])[None, None, :, None, None, None] \
        + (frag * s_strides[4])[:, None, None, None, None, None] \
        + (sb * s_strides[5])[None, None, None, None, :, None]
    sbyte = torch.gather(sflat, 0, s_off.reshape(-1)).view(n, kvh, d // 64, 2)
    sbyte = torch.where(sbyte < 0, sbyte + 256, sbyte)
    sc = torch.exp2(sbyte - 127)
    sc = torch.where(sbyte == 255, torch.ones_like(sc), sc)
    k_out = (kval.view(n, kvh, d // 64, 2, sub) * sc.unsqueeze(-1)).view(n, kvh, d)

    vflat = _bitcast(value_cache, torch.int8).index_select(0, blk_u).to(torch.float32).reshape(-1)
    vbyte = torch.gather(vflat, 0, k_off.reshape(-1))
    vval = _decode_e4m3(vbyte).view(n, kvh, d)
    v_out = vval / v_recip.to(torch.float32).view(kvh, d)

    live3 = live.view(n, 1, 1)
    return k_out * live3, v_out * live3
