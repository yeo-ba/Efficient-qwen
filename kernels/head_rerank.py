"""Experimental two-pass Triton vocabulary top-k and BF16 candidate rescoring."""
import torch
import triton
import triton.language as tl


@triton.jit
def _tile_candidates(logits, values, indices, N: tl.constexpr, TILES: tl.constexpr,
                     TOP: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    scores = tl.load(logits + row * N + offsets, offsets < N, other=-float('inf')).to(tl.float32)
    for rank in tl.static_range(TOP):
        best = tl.max(scores, 0)
        index = tl.min(tl.where((offsets < N) & (scores == best), offsets, 2147483647), 0)
        destination = (row * TILES + tile) * TOP + rank
        tl.store(values + destination, best)
        tl.store(indices + destination, index)
        scores = tl.where(offsets == index, -float('inf'), scores)


@triton.jit
def _select_and_rescore(logits, x, weight, values, indices, selected,
                        N: tl.constexpr, K: tl.constexpr, PROPOSALS: tl.constexpr,
                        TOP: tl.constexpr, P_BLOCK: tl.constexpr, K_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, P_BLOCK)
    scores = tl.load(values + row * PROPOSALS + p, p < PROPOSALS, other=-float('inf'))
    ids = tl.load(indices + row * PROPOSALS + p, p < PROPOSALS, other=2147483647)
    d = tl.arange(0, K_BLOCK)
    activation = tl.load(x + row * K + d, d < K, other=0).to(tl.float32)
    for rank in tl.static_range(TOP):
        best = tl.max(scores, 0)
        index = tl.min(tl.where(scores == best, ids, 2147483647), 0)
        valid = index < N
        original = tl.load(weight + tl.minimum(index, N - 1) * K + d,
                           (d < K) & valid, other=0).to(tl.float32)
        exact = tl.sum(activation * original, 0)
        tl.store(logits + row * N + index, exact, valid)
        tl.store(selected + row * TOP + rank, index)
        scores = tl.where(ids == index, -float('inf'), scores)


def rerank_topk(logits, x, weight, top=4, block=2048):
    if (top not in (4, 8) or block not in (1024, 2048, 4096)
            or weight.shape != (248320, 2560) or x.ndim != 2 or x.shape[1] != 2560
            or logits.shape != (x.shape[0], 248320)
            or any(t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous() for t in (logits, x, weight))):
        raise ValueError('Unsupported experimental rerank input')
    m, n = logits.shape
    tiles = triton.cdiv(n, block)
    values = torch.empty((m, tiles * top), device=logits.device, dtype=torch.float32)
    indices = torch.empty((m, tiles * top), device=logits.device, dtype=torch.int32)
    selected = torch.empty((m, top), device=logits.device, dtype=torch.int32)
    _tile_candidates[(m, tiles)](logits, values, indices, n, tiles, top, block, num_warps=4)
    _select_and_rescore[(m,)](logits, x, weight, values, indices, selected,
        n, 2560, tiles * top, top, triton.next_power_of_2(tiles * top), 4096, num_warps=4)
    return logits, selected
