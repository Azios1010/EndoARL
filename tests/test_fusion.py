import unittest

import torch

from dust3r.memory.config import FusionConfig
from dust3r.memory.fusion import HybridMemoryFusion


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.fusion = HybridMemoryFusion(FusionConfig(hidden_dim=8, output_dim=16), input_dim=16)
        self.q, self.st, self.lt = (torch.randn(1, 3, 16) for _ in range(3))

    def test_all_availability_modes(self):
        for st, lt in ((None, None), (self.st, None), (None, self.lt), (self.st, self.lt)):
            output, diag = self.fusion(self.q, st, lt)
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(output.shape, self.q.shape)
            self.assertEqual(diag.has_short, st is not None)
            self.assertEqual(diag.has_long, lt is not None)
            self.assertFalse(diag.gate_mean.requires_grad)
            if st is None and lt is None:
                torch.testing.assert_close(output, self.q)
            if st is not None and lt is None:
                torch.testing.assert_close(output, self.q + st)
                self.assertEqual(diag.gate_mean.item(), 1)
            if st is None and lt is not None:
                torch.testing.assert_close(output, self.q + lt)
                self.assertEqual(diag.gate_mean.item(), 0)

    def test_both_branches_train_gate(self):
        output, _ = self.fusion(self.q, self.st, self.lt)
        output.square().mean().backward()
        for parameter in self.fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_nonfinite_input_rejected(self):
        self.st[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.fusion(self.q, self.st, self.lt)


if __name__ == "__main__":
    unittest.main()
