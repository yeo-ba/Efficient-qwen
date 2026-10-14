#!/usr/bin/env python3
"""AdaptFM quality scores over an OpenAI-compatible local chat endpoint."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
from budget import require_budget
from latency import MemorySampler


@dataclass(frozen=True)
class Task:
    name: str
    harness: str
    fewshot: int
    metric: str
    threshold: float
    thinking: bool
    max_tokens: int


TASKS = {
    "mmlu_pro": Task("mmlu_pro", "mmlu_pro", 5, "exact_match,custom-extract", .621, False, 2048),
    "ifeval": Task("ifeval", "ifeval", 0, "inst_level_strict_acc,none", .814, False, 512),
    "gpqa_diamond": Task("gpqa_diamond", "gpqa_diamond_cot_zeroshot", 0,
                         "exact_match,flexible-extract", .630, True, 12288),
}


def extract_score(task, results):
    # Match the organizer's macro-average across MMLU-Pro subjects.
    if task.name == "mmlu_pro":
        values = [row[task.metric] for name, row in results.items()
                  if name.startswith("mmlu_pro_") and task.metric in row]
        if not values:
            raise ValueError(f"No MMLU-Pro subject metrics: {task.metric}")
        score = sum(values) / len(values)
    else:
        score = results[task.harness][task.metric]
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid score for {task.name}: {score!r}")
    return float(score)


def sample_coverage(result):
    """lm-eval logs each document once per filter; count documents, not filters."""
    total, coverage = 0, {}
    for name, records in result.get("samples", {}).items():
        documents, filters = {}, {}
        for row in records:
            doc_id, filter_name = row["doc_id"], row["filter"]
            identity = (row["doc_hash"], row["target_hash"], json.dumps(row["resps"], sort_keys=True))
            if doc_id in documents and documents[doc_id] != identity:
                raise RuntimeError(f"Conflicting filter records for {name}/{doc_id}")
            documents[doc_id] = identity
            ids = filters.setdefault(filter_name, set())
            if doc_id in ids:
                raise RuntimeError(f"Duplicate document in {name}/{filter_name}: {doc_id}")
            ids.add(doc_id)
        if not documents or any(ids != set(documents) for ids in filters.values()):
            raise RuntimeError(f"Incomplete filter coverage: {name}")
        if result.get("n-samples", {}).get(name, {}).get("effective") != len(documents):
            raise RuntimeError(f"Sample metadata disagrees with document identities: {name}")
        coverage[name] = {"unique_documents": len(documents), "logged_records": len(records),
                          "filters": {k: len(v) for k, v in filters.items()}}
        total += len(documents)
    return total, coverage


def parse_sse(lines):
    """Read OpenAI SSE, reject broken streams instead of scoring partial transport data."""
    content, reasoning = [], []
    finish, done = None, False
    for line in lines:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"Streaming server error: {event['error']}")
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta", {})
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            finish = choice.get("finish_reason") or finish
    if not done or finish is None:
        raise RuntimeError("Incomplete SSE stream: missing finish_reason or [DONE]")
    return "".join(content), "".join(reasoning), finish


def final_answer(content, reasoning, thinking):
    # The public base server has no reasoning parser: its content contains the
    # whole generated trace. GPQA flexible-extract must see that same text,
    # including reasoning when generation reaches its token ceiling.
    if thinking:
        return ((reasoning + "\n" if reasoning else "") + content).strip()
    if not thinking:
        blocks = re.findall(r"<think>(.*?)</think>", content, flags=re.S)
        if reasoning.strip() or any(x.strip() for x in blocks) or ("<think>" in content and "</think>" not in content):
            raise RuntimeError("Server generated reasoning with thinking disabled; fix its chat template")
    # Some servers expose reasoning separately; others include the think block in content.
    # The opening tag can be part of the prompt, so split on the closing tag as well.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    elif "<think>" in content:
        return ""  # Generation exhausted its budget before reaching an answer.
    return content.strip()


class ChatClient:
    def __init__(self, base_url, model, task, timeout=600, retries=2, api_key=None, audit=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.task = model, task
        self.timeout, self.retries, self.api_key = timeout, retries, api_key
        self.lock = threading.Lock()
        self.requests = self.truncated = 0
        self.failures = 0
        self.audit = audit

    def record(self, payload, attempt, error=None, answer=None, finish=None):
        with self.lock:
            self.failures += int(error is not None)
            if self.audit:
                with self.audit.open("a") as f:
                    f.write(json.dumps({"unix": time.time(), "task": self.task.name,
                        "prompt_sha256": hashlib.sha256(json.dumps(payload["messages"]).encode()).hexdigest(),
                        "attempt": attempt, "error": error, "thinking": self.task.thinking,
                        "finish_reason": finish, "generated_text": answer}) + "\n")

    def payload(self, messages, gen_kwargs):
        stop = gen_kwargs.get("until", [])
        if isinstance(stop, str):
            stop = [stop]
        stop = [s for s in (stop or []) if s]
        result = {"model": self.model, "messages": messages, "temperature": 0.0,
                  "max_tokens": self.task.max_tokens, "seed": 1234,
                  "chat_template_kwargs": {"enable_thinking": self.task.thinking},
                  "stream": self.task.thinking}
        if stop:
            result["stop"] = stop
        return result

    def generate(self, messages, gen_kwargs):
        payload = self.payload(messages, gen_kwargs)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        for attempt in range(self.retries + 1):
            require_budget()
            try:
                req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if self.task.thinking:
                        if "text/event-stream" not in response.headers.get("Content-Type", ""):
                            raise RuntimeError("GPQA requires an SSE response to stream=true")
                        content, reasoning, finish = parse_sse(response)
                    else:
                        data = json.load(response)
                        choice = data["choices"][0]
                        message = choice["message"]
                        content = message.get("content") or ""
                        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                        finish = choice.get("finish_reason")
                if finish not in {"stop", "length"}:
                    raise RuntimeError(f"Unexpected finish_reason: {finish!r}")
                answer = final_answer(content, reasoning, self.task.thinking)
                with self.lock:
                    self.requests += 1
                    self.truncated += int(finish == "length")
                self.record(payload, attempt, answer=answer, finish=finish)
                return answer
            except urllib.error.HTTPError as exc:
                self.record(payload, attempt, f"HTTP {exc.code}: {exc.reason}")
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Chat HTTP {exc.code}: {exc.read(1000).decode(errors='replace')}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                self.record(payload, attempt, str(exc))
                if attempt == self.retries:
                    raise
            except Exception as exc:
                self.record(payload, attempt, str(exc))
                raise
            time.sleep(2 ** attempt)
        raise AssertionError("Unreachable")


def make_lm(client, concurrency):
    from lm_eval.api.model import LM

    class ChatLM(LM):
        @property
        def tokenizer_name(self):
            return client.model.replace("/", "_") + f"_thinking_{client.task.thinking}"

        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            # Serialize for harness caching; the server applies the actual Qwen template.
            return json.dumps(chat_history, ensure_ascii=False)

        def generate_until(self, requests):
            def one(request):
                context, kwargs = request.args
                messages = json.loads(context)
                if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                    raise ValueError("Expected chat history from apply_chat_template")
                return client.generate(messages, kwargs)
            output = [None] * len(requests)
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(one, request): i for i, request in enumerate(requests)}
                for done, future in enumerate(as_completed(futures), 1):
                    output[futures[future]] = future.result()
                    if done == 1 or done % 10 == 0 or done == len(requests):
                        elapsed = time.monotonic() - started
                        print(f"  {client.task.name}: {done}/{len(requests)} responses, "
                              f"{elapsed:.0f}s elapsed, {done / max(elapsed, 0.001):.2f} req/s", flush=True)
            return output

        def loglikelihood(self, requests):
            raise NotImplementedError("This evaluator supports generation tasks only")

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("This evaluator supports generation tasks only")

    return ChatLM()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8080/v1", help="OpenAI API prefix, ending in /v1")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Model name advertised by the server")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--limit", type=float, default=1.0, help="Fraction of each task; 1.0 is full evaluation")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--output", type=Path, default=Path("results/quality"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse complete raw task results with identical settings")
    args = p.parse_args()
    if not 0 < args.limit <= 1 or args.concurrency < 1 or args.timeout <= 0:
        p.error("limit must be in (0,1], concurrency and timeout must be positive")
    settings = {"model": args.model, "base_url": args.base_url, "limit": args.limit,
                "concurrency": args.concurrency, "tasks": [asdict(TASKS[t]) for t in args.tasks]}
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return
    if version("lm_eval") != "0.4.11":
        raise RuntimeError("Install requirements-eval.txt: lm-eval==0.4.11 is required")
    from lm_eval import simple_evaluate
    args.output.mkdir(parents=True, exist_ok=args.resume)
    result_file = args.output / "summary.json"
    if args.resume:
        summary = json.loads(result_file.read_text())
        if any(summary["settings"].get(k) != v for k, v in settings.items()):
            raise RuntimeError("Cannot resume with different evaluation settings")
        summary.setdefault("resumptions", []).append({"unix": time.time(),
            "previous_status": summary.get("status"), "previous_error": summary.get("error")})
        summary.update(status="running", all_passed=None)
        summary.pop("error", None)
    else:
        summary = {"settings": settings, "partial_dataset": args.limit < 1, "results": {},
                   "all_passed": None, "status": "running"}
    source_hashes = {}
    source_dir = args.output / ("source-resume-" + str(time.time_ns()) if args.resume else "source")
    source_dir.mkdir()
    for filename in ["quality.py", "budget.py", "latency.py"]:
        source = (Path(__file__).resolve().parent / filename).read_bytes()
        (source_dir / filename).write_bytes(source)
        source_hashes[filename] = hashlib.sha256(source).hexdigest()
    if args.resume:
        summary["resumptions"][-1].update(source_snapshot=source_dir.name, executed_source_sha256=source_hashes)
    else:
        settings["executed_source_sha256"] = source_hashes
    result_file.write_text(json.dumps(summary, indent=2) + "\n")
    memory = MemorySampler()
    if args.resume and (args.output / "memory.json").exists():
        previous_memory = json.loads((args.output / "memory.json").read_text())
        memory.samples.extend(previous_memory["samples"])
        memory.errors.extend(previous_memory["errors"])
    memory_thread = threading.Thread(target=memory.run, daemon=True)
    memory_thread.start()
    try:
        for name in args.tasks:
            task = TASKS[name]
            print(f"{name}: thinking={task.thinking}, max_tokens={task.max_tokens}", flush=True)
            client = ChatClient(args.base_url, args.model, task, timeout=args.timeout,
                                api_key=os.environ.get("EVAL_API_KEY"), audit=args.output / "requests.jsonl")
            raw_path = args.output / f"{name}.json"
            reused = args.resume and raw_path.exists()
            if reused:
                result = json.loads(raw_path.read_text())
                audit = [json.loads(line) for line in client.audit.read_text().splitlines()]
                records = [r for r in audit if r["task"] == name]
                client.requests = sum(r.get("error") is None for r in records)
                client.truncated = sum(r.get("finish_reason") == "length" for r in records)
                client.failures = sum(r.get("error") is not None for r in records)
                print(f"Reusing saved full raw results for {name}; no new model requests", flush=True)
            else:
                result = simple_evaluate(
                    model=make_lm(client, args.concurrency), tasks=[task.harness],
                    num_fewshot=task.fewshot, batch_size=1, limit=args.limit if args.limit < 1 else None,
                    apply_chat_template=True, fewshot_as_multiturn=True,
                    gen_kwargs={"max_gen_toks": task.max_tokens, "temperature": 0.0, "do_sample": False},
                    random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
                    confirm_run_unsafe_code=True, log_samples=True,
                )
                raw_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
            score = extract_score(task, result["results"])
            sample_count, coverage = sample_coverage(result)
            expected_count = {"mmlu_pro": 12032, "ifeval": 541, "gpqa_diamond": 198}[name]
            if args.limit == 1 and sample_count != expected_count:
                raise RuntimeError(f"Full evaluation incomplete: {name} has {sample_count}, expected {expected_count}")
            if client.requests != sample_count:
                raise RuntimeError(f"Request audit does not cover every document for {name}")
            summary["results"][name] = {"score": score, "metric": task.metric,
                "threshold": task.threshold, "meets_threshold": score >= task.threshold,
                "thinking": task.thinking, "requests": client.requests, "length_limited": client.truncated,
                "failed_attempts": client.failures, "samples": sample_count, "expected_full_samples": expected_count,
                "filter_coverage": coverage, "reused_raw_results": reused,
                "executed_source_sha256": summary["results"].get(name, {}).get("executed_source_sha256",
                    summary["settings"]["executed_source_sha256"]) if reused else source_hashes}
            result_file.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{name}: {score:.4f} ({score * 100:.2f}%), threshold={task.threshold}", flush=True)
        summary["status"] = "complete"
        # A subset/sampled run must never be labeled as passing all competition quality gates.
        if args.limit == 1 and set(args.tasks) == set(TASKS):
            summary["all_passed"] = all(x["meets_threshold"] for x in summary["results"].values())
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        memory.stop.set()
        memory_thread.join(timeout=4)
        summary["max_vram_mib_sampled"] = max((x["mib"] for x in memory.samples), default=None)
        (args.output / "memory.json").write_text(json.dumps({"samples": memory.samples, "errors": memory.errors}))
        result_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {result_file}")


if __name__ == "__main__":
    main()
