from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from dust3r.training.hybrid_trainer import CHECKPOINT_SCHEMA
from dust3r.training.teacher import file_sha256
from dust3r.training.validation import certify_for_ppo, validate_legacy_baseline_report


class ValidationGateTests(unittest.TestCase):
    def test_legacy_policy_requires_identified_finite_held_out_baseline(self):
        report = dict(baseline_identity='LEGACY_OFFICIAL', split='validation', manifest_sha256='manifest',
                      original_checkpoint_hashes={'endo3r': 'released'}, sequences=[{'frame_metrics': [
                          {'point_l1_normalized': 1., 'depth_abs_rel': .1}]}])
        validate_legacy_baseline_report(report, 'released', 'manifest')
        with self.assertRaises(ValueError):
            validate_legacy_baseline_report(report, 'other-weights', 'manifest')
        report['sequences'][0]['frame_metrics'][0]['depth_abs_rel'] = float('nan')
        with self.assertRaises(ValueError):
            validate_legacy_baseline_report(report, 'released', 'manifest')

    def reports(self, checkpoint):
        report = dict(baseline_identity='NEW_HYBRID', split='validation', manifest_sha256='verified-manifest',
                      image_hw=[256, 320], scale_protocol='one_detached_first_frame_scale',
                      hybrid_checkpoint_sha256=file_sha256(checkpoint),
                      actual_backends={'short': 'sdpa_flash', 'long': 'fla_kda_chunk'},
                      sequences=[{'episode_id': 'held-out', 'frame_metrics': [
                          {'point_l1_normalized': 1., 'depth_abs_rel': .1}]}])
        baseline = deepcopy(report)
        baseline.update(baseline_identity='LEGACY_OFFICIAL', original_checkpoint_hashes={'endo3r': 'released-hash'})
        return report, baseline

    def test_certification_saves_separate_checkpoint_and_binds_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'certified.pt'
            torch.save(dict(schema=CHECKPOINT_SCHEMA, config={'memory': {'backend': 'hybrid_flash_kda'}},
                            model={'weight': torch.ones(1)}), source)
            original_hash = file_sha256(source)
            report, baseline = self.reports(source)
            summary = certify_for_ppo(source, report, baseline, output)
            self.assertTrue(summary['validated_for_ppo'])
            self.assertEqual(file_sha256(source), original_hash)
            self.assertTrue(torch.load(output, weights_only=True)['validation_summary']['validated_for_ppo'])

    def test_quality_provenance_reference_and_overwrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pt'
            output = Path(directory) / 'certified.pt'
            torch.save(dict(schema=CHECKPOINT_SCHEMA, config={'memory': {'backend': 'hybrid_flash_kda'}}), source)
            report, baseline = self.reports(source)
            with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                certify_for_ppo(source, report, baseline, source)
            for field, value in [('split', 'train'), ('manifest_sha256', None),
                                 ('actual_backends', {'short': 'math_reference', 'long': 'torch_reference'})]:
                invalid = deepcopy(report)
                invalid[field] = value
                with self.assertRaises(ValueError):
                    certify_for_ppo(source, invalid, baseline, output)
            report['sequences'][0]['frame_metrics'][0]['depth_abs_rel'] = .2
            with self.assertRaisesRegex(ValueError, 'quality gate failed'):
                certify_for_ppo(source, report, baseline, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
