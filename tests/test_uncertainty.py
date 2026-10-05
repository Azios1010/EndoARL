import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dust3r.memory.config import UncertaintyConfig
from dust3r.memory.types import MemoryCandidate
from dust3r.memory.uncertainty import UncertaintyEstimator


class StubFlow(nn.Module):
    def __init__(self, value=0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.calls = 0
        self.value = value

    def forward(self, image, ref, iters, test_mode):
        self.calls += 1
        self.last_iters = iters
        return image.new_full((image.shape[0], 2, *image.shape[-2:]), self.value)


def candidate():
    return MemoryCandidate(torch.ones(1, 4, 8), torch.ones(1, 4, 8),
        image=torch.zeros(1, 3, 4, 4, dtype=torch.uint8), pointmap=torch.ones(1, 4, 4, 3),
        valid_mask=torch.ones(1, 4, 4, dtype=torch.bool), confidence=torch.ones(1, 4, 4))


class UncertaintyTests(unittest.TestCase):
    def test_enabled_missing_checkpoint_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'No automatic download'):
            UncertaintyEstimator(UncertaintyConfig(), 'missing-test-raft.pth')

    def test_disabled_and_first_write_skip_raft(self):
        flow = StubFlow()
        disabled = UncertaintyEstimator(UncertaintyConfig(enabled=False), flow_model=flow, patch_hw=(2, 2))
        self.assertTrue(torch.equal(disabled.estimate(candidate(), candidate()).weights, torch.ones(1, 4, 1)))
        enabled = UncertaintyEstimator(UncertaintyConfig(), flow_model=flow, patch_hw=(2, 2))
        self.assertFalse(enabled.estimate(candidate(), None).raft_called)
        self.assertEqual(flow.calls, 0)

    def test_registered_frozen_eval_and_valid_geometry(self):
        flow = StubFlow()
        estimator = UncertaintyEstimator(UncertaintyConfig(), flow_model=flow, patch_hw=(2, 2))
        estimator.train()
        self.assertFalse(flow.training)
        self.assertFalse(flow.weight.requires_grad)
        output = estimator.estimate(candidate(), candidate())
        self.assertTrue(torch.equal(output.weights, torch.ones(1, 4, 1)))
        self.assertEqual(output.uncertainty_stats[0, 2], 1)
        self.assertEqual(flow.calls, 1)
        self.assertEqual(flow.last_iters, 20)
        self.assertFalse(output.weights.requires_grad)

    def test_invalid_flow_no_confidence_or_nan(self):
        for value in (float('nan'), float('inf'), 1000):
            estimator = UncertaintyEstimator(UncertaintyConfig(), flow_model=StubFlow(value), patch_hw=(2, 2))
            output = estimator.estimate(candidate(), candidate())
            self.assertTrue(torch.isfinite(output.weights).all())
            self.assertFalse(output.has_any_positive_patch)
            self.assertEqual(output.uncertainty_stats[0, 2], 0)

    def test_nan_reference_pixels_are_not_high_confidence(self):
        estimator = UncertaintyEstimator(UncertaintyConfig(), flow_model=StubFlow(), patch_hw=(2, 2))
        reference = candidate()
        reference.pointmap[:, :2, :2] = float('nan')
        output = estimator.estimate(candidate(), reference)
        self.assertEqual(output.weights[0, 0], 0)
        self.assertTrue(torch.isfinite(output.weights).all())
        self.assertTrue((output.weights[0, 1:] > 0).all())

    def test_disagreement_reduces_write_strength(self):
        estimator = UncertaintyEstimator(UncertaintyConfig(), flow_model=StubFlow(), patch_hw=(2, 2))
        current = candidate()
        current.pointmap *= 10
        output = estimator.estimate(current, candidate())
        self.assertTrue(torch.allclose(output.weights, torch.full((1, 4, 1), .05)))

    def test_neutral_partial_invalid_patch(self):
        estimator = UncertaintyEstimator(UncertaintyConfig(enabled=False), patch_hw=(2, 2))
        current = candidate()
        current.valid_mask[:, :2, :2] = False
        output = estimator.neutral(current)
        self.assertEqual(output.weights[0, 0], 0)
        self.assertEqual(output.weights.sum(), 3)


if __name__ == '__main__':
    unittest.main()
