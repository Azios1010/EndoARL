import unittest

import torch

from dust3r.memory.config import FusionConfig, LongMemoryConfig, MemoryConfig, ShortMemoryConfig, load_config
from dust3r.memory.hybrid import HybridMemory
from dust3r.memory.short_flash import FlashBackendUnavailable, exact_attention_reference


def tiny_config(backend="math_reference"):
    return MemoryConfig(input_dim=16, output_dim=16,
                        short=ShortMemoryConfig(model_dim=8, num_heads=2, head_dim=4,
                                                backend=backend, require_flash=backend in {"flash", "sdpa_flash"}),
                        long=LongMemoryConfig(backend="torch_reference", model_dim=8,
                                              num_heads=2, key_dim=4, value_dim=4),
                        fusion=FusionConfig(hidden_dim=8, output_dim=16))


class ShortMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_exact_reference_matches_sdpa(self):
        q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)
        actual = exact_attention_reference(q, k, v)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_exact_reference_is_fp32_inside_autocast(self):
        q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)
        expected = exact_attention_reference(q, k, v)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = exact_attention_reference(q, k, v)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_ring_retains_four_oldest_first(self):
        memory = HybridMemory(tiny_config())
        state = memory.reset(image_hw=(2, 2), patch_hw=(1, 2))
        for index in range(7):
            state.candidate_index = index
            x = torch.randn(1, 2, 16)
            state = memory.append_short(x, x, None, state)
        self.assertEqual(state.short_occupancy, 4)
        self.assertEqual(state.short_frame_indices.tolist(), [3, 4, 5, 6])
        self.assertEqual(state.processed_candidate_count, 7)
        output, diag = memory.read(torch.randn(1, 2, 16), state)
        self.assertEqual(tuple(output.shape), (1, 2, 16))
        self.assertTrue(diag.has_short)
        self.assertIsNone(state.short_raw_k)

    def test_cached_exact_output_matches_explicit_projection(self):
        memory = HybridMemory(tiny_config())
        state = memory.reset(image_hw=(2, 2), patch_hw=(1, 2))
        k, v, q = (torch.randn(1, 2, 16) for _ in range(3))
        state = memory.append_short(k, v, None, state)
        normalized = memory._shared_norms.apply("q", q)
        qp = memory.short.q_proj(normalized).reshape(1, 2, 2, 4).transpose(1, 2)
        expected = exact_attention_reference(qp, state.short_k.flatten(1, 2).transpose(1, 2),
                                             state.short_v.flatten(1, 2).transpose(1, 2))
        expected = memory.short.out_proj(expected.transpose(1, 2).reshape(1, 2, 8))
        torch.testing.assert_close(memory.short.read(q, state), expected)

    def test_explicit_flash_has_no_cpu_fallback(self):
        memory = HybridMemory(tiny_config("flash"))
        state = memory.reset(image_hw=(2, 2), patch_hw=(1, 2))
        x = torch.randn(1, 2, 16)
        state = memory.append_short(x, x, None, state)
        with self.assertRaisesRegex(FlashBackendUnavailable, "SERVER_VALIDATION_REQUIRED"):
            memory.read(x, state)

    def test_generic_sdpa_is_labeled_unverified(self):
        memory = HybridMemory(tiny_config("sdpa"))
        state = memory.reset(image_hw=(2, 2), patch_hw=(1, 2))
        x = torch.randn(1, 2, 16)
        state = memory.append_short(x, x, None, state)
        memory.read(x, state)
        self.assertEqual(memory.short.last_backend, "sdpa_dispatch_unverified")

    def test_autocast_cache_uses_projected_dtype(self):
        memory = HybridMemory(tiny_config())
        state = memory.reset(image_hw=(2, 2), patch_hw=(1, 2))
        x = torch.randn(1, 2, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            state = memory.append_short(x, x, None, state)
            state = memory.append_short(x, x, None, state)
            self.assertEqual(state.short_k.dtype, torch.bfloat16)
            self.assertEqual(state.short_v.dtype, torch.bfloat16)
            output, _ = memory.read(x, state)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_strict_config_rejects_legacy_fields_and_fake_flash(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_config({"memory": {"top_k": 20000}})
        with self.assertRaisesRegex(ValueError, "require_flash"):
            load_config({"memory": {"short": {"backend": "math_reference"}}})
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_config({"uncertainty": {"enabled": "false"}})


if __name__ == "__main__":
    unittest.main()
