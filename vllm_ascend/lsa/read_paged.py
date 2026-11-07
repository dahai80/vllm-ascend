import torch
import triton
import triton.language as tl

# Sparse read over paged KV pool rows. One program reads one (request row, kv head group):
# it walks the row's keep list (absolute pool slots) in blocks, gathers k/v straight from the
# flat pool [rows, KVH, DIM], and runs an online softmax in registers. Keep ids are pool rows,
# never -1 padded beyond the tail (the kernel masks the tail only).
BLOCK_C = 32
PAD_HEADS = 16
WARPS = 4


@triton.jit
def _read_pool_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    slots_ptr,
    out_ptr,
    scale,
    keep_length,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    HG: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    PQ: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // KVH
    g = pid % KVH
    d = tl.arange(0, DIM)
    h = tl.arange(0, PQ)
    hlive = h < HG

    q_tile = tl.load(
        q_ptr + (row * QH + g * HG + h[:, None]) * DIM + d[None, :],
        mask=hlive[:, None],
        other=0.0,
    )

    m = tl.full([PQ], -1e30, tl.float32)
    l = tl.zeros([PQ], tl.float32)
    acc = tl.zeros([PQ, DIM], tl.float32)
    for start in range(0, keep_length, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        live = offs < keep_length
        idx = tl.load(slots_ptr + row * keep_length + offs, mask=live, other=0)
        live = live & (idx >= 0)
        idx = tl.maximum(idx, 0)
        kt = tl.load(
            k_ptr + (idx[None, :] * KVH + g) * DIM + d[:, None],
            mask=live[None, :],
            other=0.0,
        )
        s = tl.dot(q_tile, kt) * scale
        s = tl.where(live[None, :], s, -1e30)
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(live[None, :], p, 0.0)
        vv = tl.load(
            v_ptr + (idx[:, None] * KVH + g) * DIM + d[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        l = l * alpha + tl.sum(p, 1)
        m = m_new

    l = tl.where(l > 0, l, 1.0)
    tl.store(
        out_ptr + (row * QH + g * HG + h[:, None]) * DIM + d[None, :],
        (acc / l[:, None]).to(out_ptr.dtype.element_ty),
        mask=hlive[:, None],
    )


def read_pool_rows(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slots: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # q [R, Hq, DIM] bf16; pools flat [rows, KVH, DIM]; slots [R, keep] int32 pool rows
    rows, q_heads, dim = q.shape
    kv_heads = key_pool.shape[1]
    group = q_heads // kv_heads
    block = 32 if dim <= 64 else (64 if dim <= 128 else 32)
    out = torch.empty_like(q)
    _read_pool_kernel[(rows * kv_heads,)](
        q,
        key_pool,
        value_pool,
        slots,
        out,
        scale,
        slots.shape[-1],
        QH=q_heads,
        KVH=kv_heads,
        HG=group,
        DIM=dim,
        BLOCK=block,
        PQ=PAD_HEADS,
        num_warps=WARPS,
        num_stages=3,
    )
    return out
