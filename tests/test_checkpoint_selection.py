"""Synthetic CPU artifact tests; never claim real-data quality certification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dust3r.training.hybrid_trainer import CHECKPOINT_SCHEMA
from dust3r.training.selection import select_hybrid_checkpoint, write_checkpoint_selection
from dust3r.training.teacher import file_sha256
from tools.select_hybrid_checkpoint import main


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / 'full_manifest.json'
        self.manifest.write_text('{"synthetic_unittest_only": true}', encoding='utf-8')
        self.baseline = {
            'baseline_identity': 'LEGACY_OFFICIAL', 'split': 'validation',
            'manifest_sha256': file_sha256(self.manifest), 'image_hw': [256, 320],
            'scale_protocol': 'one_detached_first_frame_scale', 'max_frames': None,
            'subset_sha256': 'a' * 64, 'evaluation_streaming': True,
            'source_provenance': {'source_tree_sha256': 'b' * 64},
            'original_checkpoint_hashes': {'dust3r': 'c' * 64, 'endo3r': 'd' * 64},
            'sequences': [self._sequence('seq_a', 2, 1., 1.), self._sequence('seq_b', 3, 1., 1.)]}
        self.baseline_path = self.root / 'baseline.json'
        self._save(self.baseline_path, self.baseline)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _sequence(episode, count, point, depth):
        return {'episode_id': episode, 'source_frame_ids': [f'source_{i}' for i in range(count)],
                'frame_metrics': [{'frame': i, 'point_l1_normalized': point,
                                   'depth_abs_rel': depth, 'valid_ratio': .5}
                                  for i in range(count)]}

    @staticmethod
    def _save(path, value):
        Path(path).write_text(json.dumps(value), encoding='utf-8')

    def _candidate(self, name, point=.9, depth=.9, steps=20):
        checkpoint = self.root / f'{name}.pt'
        torch.save({'schema': CHECKPOINT_SCHEMA,
                    'config': {'memory': {'backend': 'hybrid_flash_kda'}},
                    'global_step': steps, 'model': {'toy': torch.tensor([point, depth])},
                    'original_checkpoint_hashes': self.baseline['original_checkpoint_hashes'],
                    'source_provenance': self.baseline['source_provenance']}, checkpoint)
        report = copy.deepcopy(self.baseline)
        report.update(baseline_identity='NEW_HYBRID', hybrid_checkpoint_sha256=file_sha256(checkpoint),
                      actual_backends={'short': 'sdpa_flash', 'long': 'fla_kda_chunk'})
        for sequence in report['sequences']:
            for frame in sequence['frame_metrics']:
                frame.update(point_l1_normalized=point, depth_abs_rel=depth)
        report_path = self.root / f'{name}.json'
        self._save(report_path, report)
        return {'label': name, 'checkpoint': checkpoint, 'report': report_path}, report

    def _select(self, candidates, **kwargs):
        return select_hybrid_checkpoint(candidates, self.baseline_path, **kwargs)

    def test_readonly_ranking_recomputes_scores_and_ties_by_depth(self):
        a, _ = self._candidate('a', .8, .9)
        b, report = self._candidate('b', .8, .7)
        report['sequences'][0]['mean_metrics'] = {'point_l1_normalized': 0}  # ignored
        self._save(b['report'], report)
        before = {item.name: file_sha256(item) for item in self.root.iterdir()}
        result = self._select([a, b])
        self.assertEqual(result['selected_candidate']['label'], 'b')
        self.assertEqual(result['selected_candidate']['frame_count'], 5)
        self.assertAlmostEqual(result['selected_candidate']['score']['point_l1_normalized'], .8)
        self.assertFalse(result['writes_performed'])
        self.assertEqual(before, {item.name: file_sha256(item) for item in self.root.iterdir()})

    def test_frame_uniform_mean_not_sequence_uniform(self):
        a, report = self._candidate('a')
        for frame in report['sequences'][0]['frame_metrics']:
            frame['point_l1_normalized'] = .5
        for frame in report['sequences'][1]['frame_metrics']:
            frame['point_l1_normalized'] = 1.
        self._save(a['report'], report)
        result = self._select([a])
        self.assertAlmostEqual(result['selected_candidate']['score']['point_l1_normalized'], .8)

    def test_train_and_test_reports_never_select(self):
        for split in ('train', 'test'):
            with self.subTest(split=split):
                a, report = self._candidate(split)
                report['split'] = split
                self._save(a['report'], report)
                result = self._select([a])
                self.assertIsNone(result['selected_candidate'])
                self.assertIn('Only validation', result['candidates'][0]['errors'][0])

    def test_quality_gate_is_per_sequence_and_both_metrics(self):
        a, report = self._candidate('point_gate', .1, .1)
        report['sequences'][0]['frame_metrics'][0]['point_l1_normalized'] = 2.1
        report['sequences'][0]['frame_metrics'][1]['point_l1_normalized'] = 2.1
        self._save(a['report'], report)
        result = self._select([a])
        self.assertIn('quality gate failed', result['candidates'][0]['errors'][0])
        b, _ = self._candidate('depth_gate', .1, 1.06)
        self.assertIsNone(self._select([b])['selected_candidate'])

    def test_protocol_resolution_and_gt_coverage_mismatches(self):
        changes = {'image_hw': [320, 256], 'manifest_sha256': 'e' * 64,
                   'subset_sha256': 'f' * 64, 'scale_protocol': 'per_frame_median'}
        for field, value in changes.items():
            with self.subTest(field=field):
                a, report = self._candidate(field)
                report[field] = value
                self._save(a['report'], report)
                self.assertIsNone(self._select([a])['selected_candidate'])
        a, report = self._candidate('gt_coverage')
        report['sequences'][0]['frame_metrics'][0]['valid_ratio'] = .25
        self._save(a['report'], report)
        self.assertIn('GT coverage differs', self._select([a])['candidates'][0]['errors'][0])

    def test_source_tree_and_frame_ids_must_match(self):
        a, report = self._candidate('source')
        report['source_provenance']['source_tree_sha256'] = 'e' * 64
        self._save(a['report'], report)
        self.assertIn('source_tree_sha256 mismatch', self._select([a])['candidates'][0]['errors'][0])
        b, report = self._candidate('frame_ids')
        report['sequences'][0]['source_frame_ids'][0] = 'wrong'
        self._save(b['report'], report)
        self.assertIn('source frame IDs differ', self._select([b])['candidates'][0]['errors'][0])

    def test_missing_source_provenance_requires_regeneration(self):
        a, report = self._candidate('legacy_report')
        self.baseline.pop('source_provenance')
        self._save(self.baseline_path, self.baseline)
        result = self._select([a])
        self.assertIsNone(result['selected_candidate'])
        self.assertIn('Missing source_provenance', result['baseline_errors'][0])

    def test_checkpoint_hash_and_reference_backends_rejected(self):
        a, report = self._candidate('bad_hash')
        report['hybrid_checkpoint_sha256'] = 'e' * 64
        self._save(a['report'], report)
        self.assertIn('checkpoint SHA256', self._select([a])['candidates'][0]['errors'][0])
        b, report = self._candidate('reference')
        report['actual_backends'] = {'short': 'math_reference', 'long': 'torch_reference'}
        self._save(b['report'], report)
        self.assertIn('actual Flash/FLA', self._select([b])['candidates'][0]['errors'][0])

    def test_nonfinite_metrics_and_failures_never_hidden(self):
        cases = [('nan', float('nan')), ('inf', float('inf')), ('negative', -.1)]
        for name, value in cases:
            with self.subTest(name=name):
                a, report = self._candidate(name)
                report['sequences'][0]['frame_metrics'][0]['depth_abs_rel'] = value
                self._save(a['report'], report)
                result = self._select([a])
                self.assertIsNone(result['selected_candidate'])
                json.dumps(result, allow_nan=False)
        a, report = self._candidate('pose_failure')
        report['sequences'][0]['frame_metrics'][0]['pose_success'] = 0.
        self._save(a['report'], report)
        self.assertIn('Pose failure', self._select([a])['candidates'][0]['errors'][0])

    def test_smoke_checkpoint_budget_is_explicit_not_invented(self):
        a, _ = self._candidate('smoke', steps=2)
        self.assertIsNotNone(self._select([a])['selected_candidate'])
        result = self._select([a], min_training_steps=1000)
        self.assertIsNone(result['selected_candidate'])
        self.assertIn('smoke checkpoint excluded', result['candidates'][0]['errors'][0])
        self.assertTrue(result['minimum_steps_is_not_convergence_proof'])

    def test_bounded_validation_requires_explicit_acknowledgement(self):
        self.baseline['max_frames'] = 3
        self._save(self.baseline_path, self.baseline)
        a, _ = self._candidate('bounded')
        self.assertIsNone(self._select([a])['selected_candidate'])
        result = self._select([a], allow_bounded_validation=True)
        self.assertIsNotNone(result['selected_candidate'])
        self.assertEqual(result['coverage_scope'], 'explicit_bounded_prefix')

    def test_manifest_binding_does_not_relabel_filtered_train_gt(self):
        a, _ = self._candidate('manifest')
        self.assertIsNotNone(self._select([a], ppo_manifest=self.manifest)['selected_candidate'])
        filtered = self.root / 'training_only.json'
        filtered.write_text('{"filtered": true}', encoding='utf-8')
        result = self._select([a], ppo_manifest=filtered)
        self.assertIsNone(result['selected_candidate'])
        self.assertIn('PPO manifest SHA256 differs', result['baseline_errors'][0])

    def test_certification_writes_separate_artifacts_and_never_overwrites(self):
        a, _ = self._candidate('selected')
        selection = self._select([a], ppo_manifest=self.manifest)
        original_sha = file_sha256(a['checkpoint'])
        checkpoint, report_path = self.root / 'certified.pt', self.root / 'selection.json'
        saved = write_checkpoint_selection(selection, checkpoint, report_path)
        self.assertEqual(saved['status'], 'CERTIFIED')
        self.assertEqual(file_sha256(a['checkpoint']), original_sha)
        payload = torch.load(checkpoint, weights_only=True)
        self.assertTrue(payload['validation_summary']['validated_for_ppo'])
        self.assertEqual(json.loads(report_path.read_text())['selected_candidate']['label'], 'selected')
        self.assertEqual(saved['split'], 'validation')
        self.assertTrue(saved['ppo_certification']['validated_for_ppo'])
        self.assertEqual(saved['hybrid_checkpoint_sha256'], original_sha)
        self.assertEqual(saved['checkpoint_selection']['status'], 'SELECTED_NOT_CERTIFIED')
        with self.assertRaises(FileExistsError):
            write_checkpoint_selection(selection, checkpoint, report_path)

    def test_stale_reports_rejected_before_writes(self):
        a, report = self._candidate('stale')
        selection = self._select([a])
        report['unreviewed_change'] = True
        self._save(a['report'], report)
        with self.assertRaisesRegex(ValueError, 'reports changed'):
            write_checkpoint_selection(selection, self.root / 'certified.pt', self.root / 'selection.json')
        self.assertFalse((self.root / 'certified.pt').exists())

    def test_invalid_candidate_logged_without_hiding_good_candidate(self):
        a, _ = self._candidate('good', .8, .8)
        b, _ = self._candidate('bad', 10, 10)
        result = self._select([a, b])
        self.assertEqual(len(result['candidates']), 2)
        self.assertTrue(result['candidates'][1]['errors'])
        self.assertEqual(result['selected_candidate']['label'], 'good')

    def test_invalid_checkpoint_is_recorded_without_unsafe_deserialization(self):
        a, report = self._candidate('invalid_checkpoint')
        Path(a['checkpoint']).write_bytes(b'not a torch checkpoint')
        report['hybrid_checkpoint_sha256'] = file_sha256(a['checkpoint'])
        self._save(a['report'], report)
        result = self._select([a])
        self.assertIsNone(result['selected_candidate'])
        self.assertTrue(result['candidates'][0]['errors'])

    def test_cli_readonly_and_output_authorization(self):
        from contextlib import redirect_stdout, redirect_stderr
        import io
        a, _ = self._candidate('cli')
        arguments = ['--baseline-report', str(self.baseline_path), '--candidate', str(a['checkpoint']), str(a['report'])]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(arguments), 0)
        self.assertFalse(json.loads(output.getvalue())['writes_performed'])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([*arguments, '--output-report', str(self.root / 'report.json')])


if __name__ == '__main__':
    unittest.main()
