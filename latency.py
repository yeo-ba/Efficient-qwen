"""Single-client raw-text latency, exact token lengths, durable failures and memory samples."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
from budget import require_budget

SCENARIOS = {"short": (64, 128), "medium": (2048, 256), "long": (8192, 256)}
OFFICIAL_MS = {"short": 2582, "medium": 5441, "long": 6576}


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def exact_prompt(tokenizer, n, unique, shared=False):
    # Unshared prompts differ from their first tokens, including every warmup.
    # Shared prompts differ only in a short suffix. No benchmark data is used.
    nonce = hashlib.sha256(str(unique).encode()).hexdigest()
    filler = "The quick brown fox jumps over the lazy dog. "
    prefix = "Shared document: " if shared else nonce + " Document: "
    suffix = "\nRequest " + nonce[:12] + ": continue."
    target = n - len(tokenizer.encode(suffix, add_special_tokens=False))
    ids = tokenizer.encode(prefix + filler * (n // 6 + 20), add_special_tokens=False)
    for k in range(target - 10, target + 11):
        text = tokenizer.decode(ids[:k], skip_special_tokens=False) + suffix
        actual = tokenizer.encode(text, add_special_tokens=False)
        if len(actual) == n:
            return text, actual
    raise ValueError(f"Cannot construct an exact {n}-token raw prompt")


class MemorySampler:
    def __init__(self):
        self.samples, self.errors = [], []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, check=True, timeout=3)
                self.samples.append({"unix": time.time(), "mib": float(r.stdout.strip())})
            except Exception as e:
                self.errors.append(str(e))
            self.stop.wait(0.1)


def summarize(rows):
    good = [r for r in rows if r["valid"]]
    times = [r["latency_ms"] for r in good]
    return {"attempts": len(rows), "valid_runs": len(good), "failures": len(rows) - len(good),
            "median_ms": statistics.median(times) if times else None,
            "p95_ms": percentile(times, .95),
            "output_tokens_per_second": sum(r["actual_output_tokens"] for r in good) / (sum(times) / 1000) if times else None,
            "median_output_tokens_per_second": statistics.median([r["actual_output_tokens"] * 1000 / r["latency_ms"] for r in good]) if good else None}


def run(args):
    if urlparse(args.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Latency must run on the serving host through loopback")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(concurrency=1, temperature=0, raw_text=True, chat_template=False,
                  ignore_eos=True, min_tokens="equal to max_tokens (speed only)",
                  profiler_enabled=False, p95_definition="linear interpolation",
                  output_tps_definition="actual generated tokens / client end-to-end seconds",
                  memory_definition="whole GPU nvidia-smi memory.used, sampled every ~100ms; not allocator peak")
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    mem = MemorySampler()
    t = threading.Thread(target=mem.run, daemon=True)
    t.start()
    all_rows, summaries = [], {}
    try:
        with (args.output / "requests.jsonl").open("w", buffering=1) as raw:
            for name, (input_n, output_n) in SCENARIOS.items():
                for i in range(args.warmup + args.runs):
                    require_budget()
                    prompt, ids = exact_prompt(tokenizer, input_n, f"{name}-{i}", args.cache_workload == "shared")
                    payload = {"model": args.model, "prompt": prompt, "temperature": 0,
                               "max_tokens": output_n, "min_tokens": output_n, "ignore_eos": True,
                               "seed": 1234, "stream": False, "add_special_tokens": False,
                               "skip_special_tokens": False}
                    row = {"stage": args.stage, "scenario": name, "cache_workload": args.cache_workload,
                           "index": i, "warmup": i < args.warmup, "input_tokens": input_n,
                           "output_tokens": output_n, "actual_input_tokens": len(ids),
                           "actual_output_tokens": None, "retokenized_output_tokens": None,
                           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "valid": False, "error": None, "request_unix": time.time()}
                    started = time.perf_counter()
                    try:
                        result = post(args.base_url.rstrip("/") + "/completions", payload)
                        row["latency_ms"] = (time.perf_counter() - started) * 1000
                        row["response"] = result
                        usage = result["usage"]
                        row["server_input_tokens"] = usage["prompt_tokens"]
                        row["actual_output_tokens"] = usage["completion_tokens"]
                        choice = result["choices"][0]
                        row["finish_reason"] = choice["finish_reason"]
                        row["retokenized_output_tokens"] = len(tokenizer.encode(choice["text"], add_special_tokens=False))
                        # Retokenization can merge tokens; server usage is authoritative for generated count.
                        if usage["prompt_tokens"] != input_n or usage["completion_tokens"] != output_n:
                            raise ValueError("Token count mismatch / early termination")
                        if choice["finish_reason"] != "length":
                            raise ValueError("Unexpected finish_reason despite forced output length")
                        row["valid"] = True
                    except Exception as e:
                        row["latency_ms"] = (time.perf_counter() - started) * 1000
                        row["error"] = f"{type(e).__name__}: {e}"
                        if isinstance(e, urllib.error.HTTPError):
                            row["error_body"] = e.read().decode(errors="replace")
                    raw.write(json.dumps(row) + "\n")
                    all_rows.append(row)
                    if i == 0 or (i + 1) % 10 == 0:
                        print(f"{args.stage} {args.cache_workload} {name} {i+1}/{args.warmup+args.runs}: {row['latency_ms']:.1f}ms valid={row['valid']}", flush=True)
                measured = [r for r in all_rows if r["scenario"] == name and not r["warmup"]]
                summaries[name] = {**summarize(measured), "input_tokens": input_n, "output_tokens": output_n,
                                   "warmup_failures": sum(not r["valid"] for r in all_rows if r["scenario"] == name and r["warmup"])}
    finally:
        mem.stop.set()
        t.join(timeout=4)
        (args.output / "memory.json").write_text(json.dumps({"samples": mem.samples, "errors": mem.errors}))
        with (args.output / "requests.csv").open("w") as f:
            fields = [k for k in all_rows[0] if k != "response"] if all_rows else ["error"]
            fields = sorted(set(fields) | {k for r in all_rows for k in r if k != "response"})
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    result = {"stage": args.stage, "cache_workload": args.cache_workload, "scenarios": summaries,
              "max_vram_mib_sampled": max((r["mib"] for r in mem.samples), default=None),
              "memory_sampling_errors": len(mem.errors),
              "eligible_latency": args.warmup == 5 and args.runs == 50 and len(summaries) == 3 and all(
                  s["valid_runs"] == 50 and s["failures"] == 0 and s["warmup_failures"] == 0 for s in summaries.values())}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="efficient-qwen")
    p.add_argument("--tokenizer", default="/home/ubuntu/qwen-weights")
    p.add_argument("--stage", required=True, help="Recorded stage or experimental candidate label")
    p.add_argument("--cache-workload", choices=["unshared", "shared"], default="unshared")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.warmup < 0 or args.runs < 1:
        p.error("Invalid run count")
    run(args)
