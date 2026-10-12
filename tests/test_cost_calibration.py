import copy
import unittest

from tools.calibrate_costs import calibration_from_profile


def fixture():
    return {'schema': 'endo3r_component_profile_v1', 'pair_count': 3,
            'image_hw': [256, 320], 'hardware': {'hardware_id': 'synthetic_GPU'},
            'actual_backends': {'short': 'sdpa_flash', 'long': 'fla_kda_recurrent'},
            'config': {'uncertainty': {'enabled': True}},
            'exclusive_component_timings': {
                'decoder': {'calls': 3, 'total_ms': 30., 'mean_ms': 10.},
                'raft': {'calls': 2, 'total_ms': 40., 'mean_ms': 20.},
                'long_update': {'calls': 3, 'total_ms': 6., 'mean_ms': 2.},
                'uncertainty_geometry': {'calls': 2, 'total_ms': 2., 'mean_ms': 1.}}}


class CostCalibrationTests(unittest.TestCase):
    def test_exclusive_per_call_costs_and_baseline(self):
        result = calibration_from_profile(fixture())
        self.assertEqual(result['baseline'], 10.)
        self.assertEqual(result['raft'], 20.)
        self.assertEqual(result['kda'], 2.)
        self.assertEqual(result['geometry'], 1.)

    def test_reference_backend_rejected(self):
        profile = fixture()
        profile['actual_backends']['long'] = 'torch_reference'
        with self.assertRaisesRegex(ValueError, 'SERVER_VALIDATION_REQUIRED'):
            calibration_from_profile(profile)

    def test_missing_action_calls_never_fabricated(self):
        profile = fixture()
        profile['exclusive_component_timings']['raft']['calls'] = 0
        with self.assertRaisesRegex(ValueError, 'calls'):
            calibration_from_profile(profile)

    def test_disabled_uncertainty_legal_zero_cost(self):
        profile = fixture()
        profile['config']['uncertainty']['enabled'] = False
        del profile['exclusive_component_timings']['raft']
        del profile['exclusive_component_timings']['uncertainty_geometry']
        result = calibration_from_profile(profile)
        self.assertEqual(result['raft'], 0.)
        self.assertEqual(result['geometry'], 0.)

    def test_nonfinite_values_rejected(self):
        for component in ('decoder', 'raft', 'long_update', 'uncertainty_geometry'):
            profile = copy.deepcopy(fixture())
            profile['exclusive_component_timings'][component]['mean_ms'] = float('nan')
            profile['exclusive_component_timings'][component]['total_ms'] = float('nan')
            with self.assertRaises(ValueError):
                calibration_from_profile(profile)


if __name__ == '__main__':
    unittest.main()
