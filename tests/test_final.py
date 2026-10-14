import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import run_final
from latency import summarize

ROOT = Path(__file__).resolve().parents[1]


class FinalReleaseTests(unittest.TestCase):
    def test_launcher_changes_only_mounts_and_container_name(self):
        record = json.loads((ROOT / 'recorded-final.json').read_text())
        original = record['commands']['candidate_speed']['docker']
        actual = run_final.command(record, Path('/models'))
        changes = {i for i, (a, b) in enumerate(zip(original, actual)) if a != b}
        allowed = {original.index('--name') + 1}
        allowed.update(i + 1 for i, value in enumerate(original) if value == '-v')
        self.assertEqual(len(actual), len(original))
        self.assertEqual(changes, allowed)
        self.assertIn('--no-enable-prefix-caching', actual)
        self.assertNotIn('--enable-prefix-caching', actual)

    def test_frozen_source_validation_rejects_mismatched_hash(self):
        record = json.loads((ROOT / 'recorded-final.json').read_text())
        run_final.validate_sources(record)
        with patch('run_final.sha256', return_value='modified'):
            with self.assertRaisesRegex(ValueError, 'differs from measured'):
                run_final.validate_sources(record)

    def test_final_metrics_recompute_from_every_saved_timing(self):
        result = json.loads((ROOT / 'evidence/final-results.json').read_text())
        self.assertEqual(len(result['rows']), 6)
        for workload in ['unshared', 'shared']:
            rows = [json.loads(line) for line in (ROOT / f'evidence/latency-{workload}.jsonl').read_text().splitlines()]
            for scenario in ['short', 'medium', 'long']:
                group = [r for r in rows if r['scenario'] == scenario]
                self.assertEqual(len(group), 55)
                self.assertEqual(sum(r['warmup'] for r in group), 5)
                self.assertTrue(all(r['valid'] and r['actual_input_tokens'] == r['input_tokens']
                                    and r['actual_output_tokens'] == r['output_tokens'] for r in group))
                measured = summarize([r for r in group if not r['warmup']])
                saved = next(r for r in result['rows'] if r['scenario'] == scenario and r['cache_workload'] == workload)
                for key in ['median_ms', 'p95_ms', 'output_tokens_per_second']:
                    self.assertTrue(math.isclose(measured[key], saved[key], rel_tol=1e-10), key)
                self.assertEqual(measured['failures'], saved['failures'])
        for name, count in [('mmlu_pro', 12032), ('ifeval', 541), ('gpqa_diamond', 198)]:
            quality = result['full_quality'][name]
            self.assertEqual(quality['samples'], count)
            self.assertGreaterEqual(quality['score'], result['quality_thresholds'][name])
        main = [r for r in result['rows'] if r['cache_workload'] == 'unshared']
        actual = sum(r['speedup_vs_same_engine_bf16-engine'] for r in main) / 3
        expected = result['summary']['mean_speedup_vs_same_engine_controls']['bf16-engine']
        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12))


if __name__ == '__main__':
    unittest.main()
