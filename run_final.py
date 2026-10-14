#!/usr/bin/env python3
"""Print or start the single measured final configuration; no alternative candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
SELECTION = ROOT / 'results/output-head-fusion022-micro/head-g128-triton4-b2048.json'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_sources(record):
    from runtime_plugin.efficient_qwen_head import load_selection
    selection = load_selection(SELECTION, root=ROOT)
    evidence = record['validation']
    expected = {
        'selection_sha256': SELECTION,
        'runtime_source_sha256': ROOT / 'runtime_plugin/efficient_qwen_head.py',
        'triton_source_sha256': ROOT / 'kernels/head_rerank.py',
    }
    if evidence['passed'] is not True or any(sha256(p) != evidence[k] for k, p in expected.items()):
        raise ValueError('Final source/configuration differs from measured runtime validation')
    if selection['image_id'] != record['candidate_engine']['image_id']:
        raise ValueError('Final image differs from kernel evidence')


def validate_models(model_root):
    manifest = json.loads((ROOT / 'models.json').read_text())
    for model in manifest['models']:
        directory = model_root / Path(model['local']).name
        for item in model['files']:
            path = directory / item['name']
            if not path.is_file() or sha256(path) != item['sha256']:
                raise ValueError(f'Model/tokenizer hash mismatch or missing file: {path}')
    for item in manifest['drafter']['files']:
        path = model_root / 'qwen-dflash-qad' / item['file']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError(f'Drafter hash mismatch or missing file: {path}')


def command(record, model_root):
    cmd = list(record['commands']['candidate_speed']['docker'])
    cmd[cmd.index('--name') + 1] = 'efficient-qwen-final'
    replacements = {
        '/home/ubuntu/qwen-qad:/model:ro': str(model_root / 'qwen-qad') + ':/model:ro',
        '/home/ubuntu/qwen-weights:/original-tokenizer:ro': str(model_root / 'qwen-weights') + ':/original-tokenizer:ro',
        '/home/ubuntu/qwen-dflash-qad:/drafter:ro': str(model_root / 'qwen-dflash-qad') + ':/drafter:ro',
        '/home/ubuntu/projects/efficient-qwen/optimization:/experiment': str(ROOT) + ':/experiment',
    }
    for old in replacements:
        if cmd.count(old) != 1:
            raise ValueError(f'Recorded command mount is missing or ambiguous: {old}')
    return [replacements.get(value, value) for value in cmd]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, default=Path('/home/ubuntu'))
    p.add_argument('--start', action='store_true', help='Verify local model hashes and budget, then start Docker')
    args = p.parse_args()
    record = json.loads((ROOT / 'recorded-final.json').read_text())
    validate_sources(record)
    cmd = command(record, args.model_root.resolve())
    print(shlex.join(cmd), flush=True)
    if args.start:
        from budget import require_budget
        require_budget()
        validate_models(args.model_root.resolve())
        require_budget()
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
