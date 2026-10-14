"""Opt-in Marlin dispatch for exact shapes backed by saved GPU microbenchmarks."""
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LIBRARIES = []
_REGISTERED = False


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def measured_median(row):
    timing = row.get("timing", {})
    samples = timing.get("samples_us", [])
    if len(samples) != 50 or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError("Tile needs all 50 finite positive timing samples")
    median = statistics.median(samples)
    if not math.isclose(median, timing["median_us"], rel_tol=1e-9):
        raise ValueError("Timing summary differs from raw samples")
    return median


def artifact(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def load_selection(path, root=ROOT):
    config = json.loads(path.read_text())
    experiments = artifact(root, config["experiments_file"])
    if digest(experiments) != config["experiments_sha256"]:
        raise ValueError("Microbenchmark evidence checksum mismatch")
    rows = [json.loads(line) for line in experiments.read_text().splitlines()]
    selected = {}
    for index in config["selected_rows"]:
        row = rows[index]
        checks = row.get("correctness", [])
        if (row.get("variant") != "tuned" or row.get("status") != "passed"
                or row.get("micro_speedup_vs_native", 0) <= 1
                or not isinstance(checks, list) or len(checks) != 4
                or {c["input_multiplier"] for c in checks} != {0.0, 0.001, 1.0, 100.0}
                or not all(c.get("passed") and c.get("finite") for c in checks)):
            raise ValueError("Selected tile lacks successful numerical and timing evidence")
        native = [r for r in rows if r.get("variant") == "native" and r.get("layer") == row["layer"]
                  and all(r[k] == row[k] for k in ("m", "n", "k"))]
        if len(native) != 1 or native[0]["status"] != "passed" or not native[0]["correctness"].get("passed"):
            raise ValueError("Missing independent native/dequantization reference")
        speedup = measured_median(native[0]) / measured_median(row)
        if speedup <= 1 or not math.isclose(speedup, row["micro_speedup_vs_native"], rel_tol=1e-9):
            raise ValueError("Claimed micro speedup differs from raw measurements")
        key = (row["m"], row["n"], row["k"])
        if key in selected:
            raise ValueError("Ambiguous tile selection for the same shape")
        selected[key] = (row["tile_k"], row["tile_n"], row["sm_count"])
    if not selected:
        raise ValueError("No validated tile selected")
    extension = artifact(root, config["extension"])
    if digest(extension) != config["extension_sha256"]:
        raise ValueError("Extension checksum mismatch")
    manifest = json.loads(extension.with_name("manifest.json").read_text())
    if manifest["vllm"] != config["vllm_version"]:
        raise ValueError("Extension ABI does not match selection")
    micro = json.loads(experiments.with_name("config.json").read_text())
    if (micro["vllm"] != config["vllm_version"]
            or micro["extension_sha256"] != config["extension_sha256"]
            or micro["profile_server"]["image_id"] != config["image_id"]):
        raise ValueError("Selection, microbenchmark and profile describe different runtimes")
    return config, selected, extension


def register():
    global _REGISTERED
    location = os.environ.get("EQC_MARLIN_TILE_CONFIG")
    if not location or _REGISTERED:
        return
    config, selection, extension = load_selection(Path(location))
    import torch
    import vllm
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    if vllm.__version__ != config["vllm_version"]:
        raise RuntimeError("Serving engine differs from the validated kernel runtime")
    torch.ops.load_library(str(extension))
    native = torch.ops._C.marlin_gemm.default
    definition = torch.library.Library("efficientqwen_dispatch", "DEF")
    definition.define(str(native._schema).split("::", 1)[1])
    implementation = torch.library.Library("efficientqwen_dispatch", "IMPL", "CUDA")
    logged = set()

    def dispatch(a, c, weight, bias, scales, a_scales, global_scale, zeros, g_idx, perm,
                 workspace, type_id, m, n, k, is_k_full=True, use_atomic_add=False,
                 use_fp32_reduce=False, is_zp_float=False):
        args = (a, c, weight, bias, scales, a_scales, global_scale, zeros, g_idx, perm,
                workspace, type_id, m, n, k, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float)
        tile = selection.get((m, n, k))
        eligible = (tile is not None and type_id == scalar_types.uint4b8.id
            and a.dtype == torch.bfloat16 and scales.dtype == torch.bfloat16
            and scales.ndim == 2 and scales.shape[0] * 32 == k
            and c is None and bias is None and a_scales is None and global_scale is None
            and all(x is None or x.numel() == 0 for x in (zeros, g_idx, perm))
            and is_k_full and not use_atomic_add and use_fp32_reduce and not is_zp_float)
        if not eligible:
            return native(*args)
        if (m, n, k) not in logged:
            # Records dispatch/capture, not replay counts or a performance claim.
            print(f"[efficient-qwen kernel] validated shape={(m, n, k)} tile={tile}", flush=True)
            logged.add((m, n, k))
        return torch.ops.efficientqwen_tuning.marlin_gemm(*args, *tile)

    implementation.impl("marlin_gemm", dispatch)
    torch.library.register_fake("efficientqwen_dispatch::marlin_gemm")(ops._marlin_gemm_fake)

    def marlin_gemm(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale,
                    b_zeros, g_idx, perm, workspace, b_q_type, size_m, size_n, size_k,
                    is_k_full=True, use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False):
        return torch.ops.efficientqwen_dispatch.marlin_gemm(
            a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros,
            g_idx, perm, workspace, b_q_type.id, size_m, size_n, size_k,
            is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float)

    _LIBRARIES.extend((definition, implementation))
    ops.marlin_gemm = marlin_gemm
    _REGISTERED = True
    print(f"[efficient-qwen kernel] loaded {len(selection)} validated shapes; config_sha256={digest(Path(location))}", flush=True)
