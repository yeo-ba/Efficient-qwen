# Reproduction notes

## Runtime

One NVIDIA A10G, driver 580.105.08. Final engine: vLLM 0.22.1, torch 2.11.0+cu130, Triton 3.6.0, Transformers 5.10.2, compressed-tensors 0.15.0.1. The measured commands and immutable image identity are in `recorded-final.json`.

The final launcher is a publication wrapper around that recorded command. It changes only host mount paths and the container name. Its CPU checks do not establish performance or quality on a new image.

## Rebuild the pinned upstream image

From this repository root, create a fresh build directory (the commands intentionally fail if it already exists):

```bash
mkdir build-final
git clone https://github.com/nota-github/adaptfm-quant-dflash.git build-final/upstream
git -C build-final/upstream checkout --detach 9454195aea3c3fec5826914ae7d34733e1bb83a0
cp -R build-final/upstream/src/vllm_plugin build-final/plugin
cp build-final/upstream/LICENSE build-final/plugin/LICENSE
cp docker/Dockerfile.dflash build-final/Dockerfile
docker build --tag efficient-qwen-final:v0.22.1 build-final
```

`Dockerfile.dflash` pins the base image by registry digest. The original experiment additionally recorded hashes of the plugin build inputs. A new local image ID is not automatically equivalent to the measured image: preserve the old evidence and run fresh GPU validation and evaluation rather than replacing old image/hash fields to make checks pass. The original microbenchmark tools and complete experimental history are in the private project backup.

## Rental ledger

Before any paid run, create `results/budget.json` with the actual new session start time and rental price. Required fields are `started_utc` (ISO UTC), `hourly_usd`, `prior_cost_usd`, `cap_usd`, `stop_work_at_usd` and `max_hours`. Account for prior costs. Use limits below the remaining $50/48-hour allowance, including time to stop the instance. `python3 budget.py --check` prints the estimate and rejects an exhausted ledger. Stopping Docker does not stop instance billing.

## Evaluation

Install the client dependencies in an isolated environment: `lm-eval[api]==0.4.11`, its task dependencies (including `langdetect` and `immutabledict`), and Transformers with support for the pinned tokenizer. Obtain authorized dataset access through the original dataset repositories. Full raw quality records are private; the public audit is a summary of those records.

After the final server is ready, the corresponding client commands are:

```bash
python3 latency.py --stage final --tokenizer /path/to/models/qwen-weights \
  --cache-workload unshared --output results/new-final-latency-unshared
python3 latency.py --stage final --tokenizer /path/to/models/qwen-weights \
  --cache-workload shared --output results/new-final-latency-shared
python3 quality.py --base-url http://localhost:8080/v1 --model efficient-qwen \
  --limit 1.0 --concurrency 8 --output results/new-final-quality
```

Use new output directories and retain errors. Quality caps are 2,048/512/12,288 for MMLU-Pro/IFEval/GPQA-Diamond, with thinking OFF/OFF/ON. A sample run does not qualify as full evaluation. Do not apply latency-only output forcing to quality. The speedup headline requires a same-machine BF16 reference with identical generation and timing settings; published ratios are historical measurements, not guarantees for a new environment.

The archived `candidate_speed` client commands also refer to historical cache-correctness checks. Those complete research tools are in the private backup; this final-only repository includes serving, speed/quality clients and final evidence.
