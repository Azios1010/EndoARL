import unittest

import torch

from dust3r.memory.config import FusionConfig, LongMemoryConfig, MemoryConfig, ShortMemoryConfig
from dust3r.memory.hybrid import HybridMemory
from dust3r.memory.types import MemoryCandidate, SharedNorms


def tiny_config():
    return MemoryConfig(input_dim=16, output_dim=16,
                        short=ShortMemoryConfig(model_dim=8, num_heads=2, head_dim=4,
                                                backend="math_reference", require_flash=False),
                        long=LongMemoryConfig(backend="torch_reference", model_dim=8,
                                              num_heads=2, key_dim=4, value_dim=4),
                        fusion=FusionConfig(hidden_dim=8, output_dim=16))


class HybridGradientTests(unittest.TestCase):
    def test_shared_norms_have_single_owner(self):
        owner = torch.nn.Module()
        owner.norm_q = torch.nn.LayerNorm(16)
        owner.norm_k = torch.nn.LayerNorm(16)
        owner.norm_v = torch.nn.LayerNorm(16)
        owner.memory = HybridMemory(tiny_config(), SharedNorms(owner.norm_q, owner.norm_k, owner.norm_v))
        keys = list(owner.state_dict())
        self.assertFalse(any(key.startswith("memory.norms") or ".norms." in key for key in keys))
        self.assertEqual(sum(key.endswith("norm_q.weight") for key in keys), 1)

    def test_full_memory_gradients_and_tbptt_refresh(self):
        torch.manual_seed(5)
        memory = HybridMemory(tiny_config())
        state = memory.reset(image_hw=(2, 2), patch_hw=(1, 2), training=True)
        key, value = torch.randn(1, 2, 16, requires_grad=True), torch.randn(1, 2, 16, requires_grad=True)
        state = memory.append_short(key, value, None, state)
        state, _ = memory.write_long(MemoryCandidate(key, value), torch.ones(1, 2, 1), state)
        output, _ = memory.read(torch.randn(1, 2, 16), state)
        output.square().mean().backward()
        for name, parameter in memory.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer = torch.optim.AdamW(memory.parameters(), lr=.01)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        detached = memory.detach(state, reproject=True)
        self.assertIsNone(detached.kda_state.grad_fn)
        self.assertIsNone(detached.short_raw_k.grad_fn)
        self.assertIsNotNone(detached.short_k.grad_fn)
        output, _ = memory.read(torch.randn(1, 2, 16), detached)
        output.square().mean().backward()
        self.assertIsNotNone(memory.short.k_proj.weight.grad)
        self.assertIsNone(memory.long.beta_proj.weight.grad)

    def test_skip_state_and_reset_do_not_leak(self):
        memory = HybridMemory(tiny_config())
        state = memory.reset(image_hw=(2, 2), patch_hw=(1, 2), sequence_id="first")
        key = torch.randn(1, 2, 16)
        state, _ = memory.write_long(MemoryCandidate(key, key), torch.ones(1, 2, 1), state)
        original_state = state.kda_state.clone()
        skipped = memory.append_short(key, key, None, state)
        self.assertTrue(torch.equal(original_state, skipped.kda_state))
        self.assertIs(skipped.kda_state, state.kda_state)
        reset = memory.reset(image_hw=(2, 2), patch_hw=(1, 2), sequence_id="second")
        self.assertFalse(reset.has_long_memory)
        self.assertEqual(reset.short_occupancy, 0)
        self.assertTrue(torch.equal(reset.kda_state, torch.zeros_like(reset.kda_state)))


if __name__ == "__main__":
    unittest.main()
