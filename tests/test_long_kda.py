import unittest
from unittest.mock import patch

import torch

from dust3r.memory.config import LongMemoryConfig
from dust3r.memory.kda_adapter import KDAAdapter, KDABackendUnavailable, kda_reference
from dust3r.memory.long_kda import LongTermKDAMemory


class LongMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.memory = LongTermKDAMemory(LongMemoryConfig(backend="torch_reference", model_dim=8,
                                                         num_heads=2, key_dim=4, value_dim=4), input_dim=16)

    def test_zero_reset_and_bounded_writes(self):
        state = self.memory.reset(1, "cpu")
        self.assertEqual(tuple(state.shape), (1, 2, 4, 4))
        self.assertEqual(state.dtype, torch.float32)
        size = state.numel()
        with torch.no_grad():
            for _ in range(25):
                k, v = torch.randn(1, 3, 16), torch.randn(1, 3, 16)
                state, diag = self.memory.update(k, v, torch.ones(1, 3, 1), state)
                self.assertTrue(diag.update_called)
                self.assertEqual(state.numel(), size)
        self.assertTrue(torch.isfinite(state).all())
        self.assertFalse(torch.equal(state, self.memory.reset(1, "cpu")))

    def test_zero_confidence_bypasses_adapter_and_preserves_identity(self):
        state = self.memory.reset(1, "cpu")
        with patch.object(self.memory.adapter, "update", side_effect=AssertionError("must not call")):
            result, diag = self.memory.update(torch.randn(1, 3, 16), torch.randn(1, 3, 16),
                                              torch.zeros(1, 3, 1), state)
        self.assertIs(result, state)
        self.assertFalse(diag.success)

    def test_read_does_not_update_or_decay(self):
        state = torch.randn(1, 2, 4, 4)
        before = state.clone()
        output = self.memory.read(torch.randn(1, 3, 16), state)
        self.assertEqual(tuple(output.shape), (1, 3, 16))
        self.assertTrue(torch.equal(before, state))

    def test_uncertainty_ablation_preserves_invalid_patch_skip(self):
        k, v = torch.randn(1, 3, 16), torch.randn(1, 3, 16)
        state = self.memory.reset(1, 'cpu')
        low = torch.tensor([[[.2], [0.], [.5]]])
        gated, _ = self.memory.update(k, v, low, state)
        self.memory.config.uncertainty_write_gate = False
        ungated, _ = self.memory.update(k, v, low, state)
        neutral, _ = self.memory.update(k, v, (low > 0).float(), state)
        self.assertFalse(torch.allclose(gated, ungated))
        torch.testing.assert_close(ungated, neutral, atol=0, rtol=0)

    def test_one_patch_delta_equation_and_zero_gates(self):
        state = torch.randn(1, 2, 4, 4)
        key = torch.nn.functional.normalize(torch.randn(1, 1, 2, 4), dim=-1)
        value, g, beta = torch.randn(1, 1, 2, 4), torch.full((1, 1, 2, 4), -.1), torch.full((1, 1, 2), .3)
        q = torch.randn_like(key)
        output, actual = kda_reference(q, key, value, g, beta, state)
        decayed = state * g[:, 0].exp().unsqueeze(-1)
        correction = .3 * (value[:, 0] - torch.einsum("bhk,bhkv->bhv", key[:, 0], decayed))
        expected = decayed + key[:, 0].unsqueeze(-1) * correction.unsqueeze(-2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(output[:, 0], torch.einsum("bhk,bhkv->bhv", q[:, 0]/2, expected))
        _, unchanged = kda_reference(q, key, value, torch.zeros_like(g), torch.zeros_like(beta), state)
        self.assertTrue(torch.equal(state, unchanged))

    def test_frame_split_equivalence_and_order(self):
        q, k, v = (torch.randn(1, 7, 2, 4) for _ in range(3))
        k = torch.nn.functional.normalize(k, dim=-1)
        g, beta = -torch.rand_like(k) * .03, torch.rand(1, 7, 2)
        state = self.memory.reset(1, "cpu")
        _, whole = kda_reference(q, k, v, g, beta, state)
        _, first = kda_reference(q[:, :3], k[:, :3], v[:, :3], g[:, :3], beta[:, :3], state)
        _, second = kda_reference(q[:, 3:], k[:, 3:], v[:, 3:], g[:, 3:], beta[:, 3:], first)
        torch.testing.assert_close(whole, second, atol=0, rtol=0)
        _, reversed_state = kda_reference(q.flip(1), k.flip(1), v.flip(1), g.flip(1), beta.flip(1), state)
        self.assertFalse(torch.allclose(whole, reversed_state))

    def test_final_state_only_backward(self):
        state = self.memory.reset(1, "cpu").requires_grad_()
        new_state, _ = self.memory.update(torch.randn(1, 3, 16), torch.randn(1, 3, 16),
                                          torch.ones(1, 3, 1), state)
        loss = self.memory.read(torch.randn(1, 3, 16), new_state).square().sum()
        loss.backward()
        for name, parameter in self.memory.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertIsNotNone(state.grad)

    def test_reference_stays_fp32_under_autocast(self):
        q, k, v = (torch.randn(1, 3, 2, 4) for _ in range(3))
        g, beta = -torch.rand_like(k)*.03, torch.rand(1, 3, 2)
        expected_output, expected_state = kda_reference(q, k, v, g, beta)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, state = kda_reference(q, k, v, g, beta)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(state.dtype, torch.float32)
        torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
        torch.testing.assert_close(state, expected_state, atol=0, rtol=0)

    def test_fla_import_is_lazy_and_missing_error_is_clear(self):
        adapter = KDAAdapter("fla_kda")
        with patch("dust3r.memory.kda_adapter.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(KDABackendUnavailable, "SERVER_VALIDATION_REQUIRED"):
                adapter.validate_backend()


if __name__ == "__main__":
    unittest.main()
