"""Opt-in W8 output projection with optional original-BF16 candidate rescoring.

This changes the output approximation and requires full model quality evaluation.
The tied input embedding remains the original BF16 tensor.
"""
import json
import os
from pathlib import Path

try:
    from .efficient_qwen_kernel import ROOT, artifact, digest, measured_median
except ImportError:
    from efficient_qwen_kernel import ROOT, artifact, digest, measured_median

_CACHE = {}
_REGISTERED = False


def load_selection(path, root=ROOT):
    config = json.loads(Path(path).read_text())
    fused = config['mode'] in {'int8_triton_top4_bf16_rerank', 'int8_triton_top8_bf16_rerank'}
    if config['mode'] not in {'int8_argmax', 'int8_top32_bf16_rerank',
                              'int8_triton_top4_bf16_rerank', 'int8_triton_top8_bf16_rerank'} or config['group_size'] not in {32, 128}:
        raise ValueError('Unsupported output-head candidate')
    raw = artifact(root, config['experiments_file'])
    if digest(raw) != config['experiments_sha256']:
        raise ValueError('Output-head experiment checksum mismatch')
    micro = json.loads(raw.with_name('config.json').read_text())
    if (micro['vllm'] != config['vllm_version'] or micro['profile_server']['image_id'] != config['image_id']
            or micro['checkpoint_sha256'] != config['checkpoint_sha256'] or micro['bits'] != 8
            or micro['weight_key'] != 'model.language_model.embed_tokens.weight'
            or 'full INT8 logits' not in micro.get('rerank_output', '')):
        raise ValueError('Selection and measured runtime/weight/algorithm differ')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if fused and (config.get('block') not in {1024, 2048, 4096}
                  or config.get('triton_source_sha256') != micro.get('triton_source_sha256')
                  or config.get('triton_source_sha256') != digest(artifact(root, 'kernels/head_rerank.py'))):
        raise ValueError('Fused-head tile or measured Triton implementation differs')
    selected = [r for r in rows if r.get('group_size') == config['group_size'] and r.get('variant') == config['mode']
                and (not fused or r.get('block') == config['block'])]
    expected = {1, 4, 8, 15, 16, 32, 64, 120, 128}
    if len(selected) != len(expected) or {r['m'] for r in selected} != expected:
        raise ValueError('Missing measured decode/draft/concurrency shapes')
    for row in selected:
        if row['status'] != 'passed' or row['validation']['passed'] is not True:
            raise ValueError('Head kernel lacks a successful independent reference')
        if fused:
            cases = row['validation'].get('independent_cases', [])
            if (len(cases) != 4 or {c['multiplier'] for c in cases} != {0, .001, 1, 100}
                    or not all(c['indices_match_independent_sort'] and c['scores_match_independent_fp32']
                               and c['unselected_logits_unchanged'] for c in cases)):
                raise ValueError('Fused-head candidate lacks independent selection/rescoring tests')
        measured_median(row)
        baseline = [r for r in rows if r.get('group_size') == config['group_size']
                    and r.get('variant') == 'bf16_argmax' and r['m'] == row['m']]
        if len(baseline) != 1 or baseline[0]['status'] != 'passed':
            raise ValueError('Missing same-input BF16 baseline')
        speedup = measured_median(baseline[0]) / measured_median(row)
        if row['m'] in {1, 15, 16} and speedup <= 1:
            raise ValueError('Observed single-request head shapes did not improve')
    return config


def pack_weight(weight, group):
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    if weight.shape != (248320, 2560) or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        raise ValueError('Expected the original tied BF16 Qwen3.5-4B output weight')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Prepare the W8 head during warmup, before CUDA graph capture')
    n, k = weight.shape
    packed = torch.empty(n, k // 4, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group, dtype=torch.bfloat16, device=weight.device)
    for start in range(0, n, 1024):
        w = weight[start:start + 1024].float().reshape(-1, k // group, group)
        scale = (w.abs().amax(-1) / 127).clamp_min(1e-10).to(torch.bfloat16)
        q = (w / scale.float().unsqueeze(-1)).round().clamp(-128, 127).to(torch.int32)
        scales[start:start + 1024] = scale
        u = (q.reshape(-1, k) + 128).reshape(-1, k // 4, 4).to(torch.int64)
        packed[start:start + 1024] = (u << (torch.arange(4, device=weight.device) * 8)).sum(-1).to(torch.int32)
    empty = torch.empty(0, dtype=torch.int32, device=weight.device)
    repacked = ops.gptq_marlin_repack(packed.t().contiguous(), empty, k, n, 8, False)
    permuted = marlin_permute_scales(scales.t().contiguous(), k, n, group, False)
    return {'weight': repacked, 'scales': permuted, 'empty': empty,
            'workspace': torch.zeros(torch.cuda.get_device_properties(weight.device).multi_processor_count,
                                     dtype=torch.int32, device=weight.device)}


def project(x, weight, buffers, mode, bias=None, *, rerank_block=2048):
    import torch
    from vllm.scalar_type import scalar_types
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError('Output-head input must be a CUDA BF16 matrix')
    x = x.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    empty = buffers['empty']
    logits = torch.ops._C.marlin_gemm(x, None, buffers['weight'], None, buffers['scales'],
        None, None, empty, empty, empty, buffers['workspace'], scalar_types.uint8b128.id,
        m, n, k, True, False, True, False)
    if bias is not None:
        logits.add_(bias)
    if mode == 'int8_top32_bf16_rerank':
        indices = logits.topk(32, dim=-1, sorted=False).indices
        original_rows = weight[indices]
        original_scores = torch.bmm(original_rows, x.unsqueeze(-1)).squeeze(-1)
        if bias is not None:
            original_scores.add_(bias[indices])
        logits.scatter_(-1, indices, original_scores)
    elif mode in {'int8_triton_top4_bf16_rerank', 'int8_triton_top8_bf16_rerank'}:
        if bias is not None:
            raise ValueError('The fused kernel is validated for the unbiased tied Qwen head')
        from kernels.head_rerank import rerank_topk
        top = 4 if mode == 'int8_triton_top4_bf16_rerank' else 8
        logits, _ = rerank_topk(logits, x, weight, top=top, block=rerank_block)
    return logits


def register():
    global _REGISTERED
    filename = os.environ.get('EQC_OUTPUT_HEAD_CONFIG')
    if not filename or _REGISTERED:
        return
    config = load_selection(Path(filename))
    import torch
    import vllm
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    if vllm.__version__ != config['vllm_version']:
        raise RuntimeError('Output-head selection was measured on another vLLM version')
    original = UnquantizedEmbeddingMethod.apply

    def apply(self, layer, x, bias=None):
        weight = layer.weight
        if tuple(weight.shape) != (248320, 2560):
            return original(self, layer, x, bias)
        key = (weight.device, weight.data_ptr(), config['group_size'])
        if key not in _CACHE:
            props = torch.cuda.get_device_properties(weight.device)
            if (props.major, props.minor) != (8, 6) or 'A10' not in props.name:
                raise RuntimeError('Output-head candidate is validated only on A10G')
            _CACHE[key] = pack_weight(weight, config['group_size'])
            # Keep a reference so a freed tensor address can never alias a new model.
            _CACHE[key]['original_weight'] = weight
            print(f'[efficient_qwen_head] W8 group={config["group_size"]}, mode={config["mode"]}; original BF16 embedding retained', flush=True)
        return project(x, weight, _CACHE[key], config['mode'], bias, rerank_block=config.get('block', 2048))

    UnquantizedEmbeddingMethod.apply = apply
    _REGISTERED = True
