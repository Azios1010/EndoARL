"""CPU-only checks for supervision, strict provenance, freeze/groups and TBPTT."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dust3r.training.losses import EpisodeScale, ReconstructionLoss, feature_distillation_loss
from dust3r.training.sequence_dataset import SequenceDataset, SequenceSample, backproject_common_frame
from dust3r.training.hybrid_trainer import (HybridTrainer, configure_trainable_parameters,
                                          build_optimizer_groups, save_training_checkpoint,
                                          load_training_checkpoint)
from dust3r.training.teacher import TeacherTraceCache, TeacherTraceMetadata, warm_start_weight


class LossTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.ones(1, 2, 2, 3)
        self.mask = torch.ones(1, 2, 2, dtype=torch.bool)

    def test_first_scale_detached_and_fixed(self):
        prediction = (self.target * 2).requires_grad_()
        scale = EpisodeScale.fit(prediction, self.target, self.mask)
        self.assertFalse(scale.scale.requires_grad)
        self.assertAlmostEqual(float(scale.scale), 0.5)
        criterion = ReconstructionLoss()
        loss, _ = criterion.frame({"pts3d": prediction * 1.2}, self.target, self.mask, scale)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_prediction_cannot_shrink_true_mask(self):
        prediction = self.target.clone()
        prediction[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            EpisodeScale.fit(prediction, self.target, self.mask)
        with self.assertRaises(ValueError):
            EpisodeScale.fit(self.target, self.target, torch.zeros_like(self.mask))

    def test_nan_gt_outside_real_mask_does_not_poison_gradient(self):
        mask = self.mask.clone()
        mask[0, 0, 0] = False
        target = self.target.clone()
        target[0, 0, 0] = float("nan")
        prediction = self.target.clone().requires_grad_()
        scale = EpisodeScale.fit(prediction, target, mask)
        loss, _ = ReconstructionLoss().frame({"pts3d": prediction * 1.2}, target, mask, scale)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_both_pair_outputs_supervised(self):
        scale = EpisodeScale.fit(self.target, self.target, self.mask)
        target = {"pointmap": self.target, "valid_mask": self.mask}
        criterion = ReconstructionLoss()
        loss, _ = criterion({"pts3d": self.target}, {"pts3d_in_other_view": 2 * self.target}, target, target, scale)
        self.assertAlmostEqual(float(loss), 0.5 / (3 ** 0.5), places=5)

    def test_distillation_detaches_teacher(self):
        student = torch.ones(1, 4, 8, requires_grad=True)
        teacher = torch.full_like(student, 2, requires_grad=True)
        loss = feature_distillation_loss(student, teacher)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertEqual(warm_start_weight(100, 1000), 0)


class DatasetTests(unittest.TestCase):
    def test_calibrated_common_coordinate_transform(self):
        depth = torch.ones(2, 2)
        pose = torch.eye(4)
        pose[0, 3] = 2
        points = backproject_common_frame(depth, torch.eye(3), pose, torch.eye(4))
        self.assertTrue(torch.allclose(points[0, 0], torch.tensor([2., 0., 1.])))
        self.assertTrue(torch.allclose(points[1, 1], torch.tensor([3., 1., 1.])))

    def _manifest(self, directory):
        import numpy as np
        frames = []
        for index in range(2):
            for key, array in (("rgb", np.zeros((2, 2, 3), dtype=np.uint8)),
                               ("depth", np.ones((2, 2), dtype=np.float32)),
                               ("valid_mask", np.ones((2, 2), dtype=np.uint8))):
                np.save(Path(directory) / f"{key}{index}.npy", array)
            frames.append({"rgb": f"rgb{index}.npy", "depth": f"depth{index}.npy",
                           "valid_mask": f"valid_mask{index}.npy", "depth_scale": 1,
                           "intrinsics": torch.eye(3).tolist(), "camera_to_world": torch.eye(4).tolist()})
        provenance = {"verified": True, "source": "synthetic calibrated unittest scene",
                      "depth_or_pointmap_gt": True, "calibration_gt": True, "pose_gt": True, "units": "metres"}
        payload = {"schema_version": 1, "sequences": [{"id": "seq", "patient_id": "test",
                    "split": "train", "provenance": provenance, "frames": frames}]}
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, payload

    def test_strict_manifest_has_no_placeholder_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self._manifest(directory)
            sample = SequenceDataset(path)[0]
            self.assertEqual(sample.frames[0]["img"].shape, (1, 3, 2, 2))
            self.assertEqual(sample.targets[0]["pointmap"].shape, (1, 2, 2, 3))
            self.assertTrue(sample.targets[0]["valid_mask"].all())
            del payload["sequences"][0]["provenance"]["verified"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                SequenceDataset(path)

    def test_patient_split_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self._manifest(directory)
            second = {**payload["sequences"][0], "id": "seq2", "split": "validation"}
            payload["sequences"].append(second)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leaks"):
                SequenceDataset(path)

    def test_empty_gt_mask_rejected(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._manifest(directory)
            np.save(Path(directory) / "valid_mask0.npy", np.zeros((2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "coverage"):
                SequenceDataset(path)[0]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybrid_memory = nn.Linear(1, 1)
        self.norm_q = nn.LayerNorm(1)
        self.value_encoder = nn.Linear(1, 1)
        self.dust3r = nn.Module()
        self.dust3r.enc_blocks = nn.ModuleList([nn.Linear(1, 1)])
        self.dust3r.dec_blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        self.dust3r.dec_blocks2 = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])


class TrainingTests(unittest.TestCase):
    def test_freezing_and_learning_rate_hierarchy(self):
        model = ToyModel()
        status = configure_trainable_parameters(model, "t1")
        self.assertTrue(all(name.startswith("hybrid_memory.") for name in status["trainable"]))
        groups = build_optimizer_groups(model)
        self.assertEqual({group["lr"] for group in groups}, {1e-4})
        configure_trainable_parameters(model, "t2", unfreeze_decoder=True)
        groups = build_optimizer_groups(model)
        self.assertEqual({group["lr"] for group in groups}, {1e-4, 1e-5, 5e-6})
        self.assertFalse(model.dust3r.enc_blocks[0].weight.requires_grad)
        self.assertFalse(model.dust3r.dec_blocks[0].weight.requires_grad)
        self.assertTrue(model.dust3r.dec_blocks[-1].weight.requires_grad)

    def test_checkpoint_roundtrip_and_backend_guard(self):
        model = ToyModel()
        config = {"memory": {"backend": "hybrid_flash_kda"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_training_checkpoint(path, model, config)
            clone = ToyModel()
            payload = load_training_checkpoint(path, clone)
            self.assertEqual(payload["schema_version"], 1)
            for original, loaded in zip(model.parameters(), clone.parameters()):
                self.assertTrue(torch.equal(original, loaded))
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, clone, "legacy")

    def test_tbptt_consumes_graph_before_optimizer_and_refresh(self):
        model = ToyModel()
        configure_trainable_parameters(model, "t1")
        class Runner:
            def reset(self, **kwargs):
                self.index = -1
                self.cache = None
                self.detach_calls = []
            def step(self, frame, terminal=False):
                self.index += 1
                if self.index == 0:
                    return None
                value = model.hybrid_memory(torch.ones(1, 1)).sigmoid() + 1
                if self.cache is not None:
                    value = value + self.cache * .1
                self.cache = value
                points = value.reshape(1, 1, 1, 1).expand(1, 2, 2, 3)
                return SimpleNamespace(candidate_index=self.index - 1, arrival_index=self.index,
                                       res1={"pts3d": points}, res2={"pts3d": points + .1}, memory_feature=points)
            def detach_state(self, refresh_short=True):
                self.detach_calls.append((self.index, refresh_short))
                self.cache = self.cache.detach()
        runner = Runner()
        targets = tuple({"pointmap": torch.ones(1, 2, 2, 3),
                         "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool)} for _ in range(9))
        sample = SequenceSample("toy", "test", "train", ({},) * 9, targets, {})
        trainer = HybridTrainer(model, runner, {"training": {"tbptt_pairs": 4}}, total_steps=10)
        logs = trainer.train_sequence(sample)
        self.assertEqual(len(logs), 2)
        self.assertEqual(runner.detach_calls, [(4, True), (8, True)])
        self.assertEqual(trainer.global_step, 2)

    def test_teacher_provenance_and_prefix_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = TeacherTraceMetadata("video", "checkpoint", "crop", "trajectory")
            cache = TeacherTraceCache(directory, metadata)
            cache.save(1, 2, torch.ones(1, 4, 8))
            self.assertEqual(cache.load(1, 2).shape, (1, 4, 8))
            with self.assertRaises(ValueError):
                cache.load(1, 3)
            altered = TeacherTraceCache(directory, TeacherTraceMetadata("video", "wrong", "crop", "trajectory"))
            with self.assertRaises(ValueError):
                altered.load(1, 2)


if __name__ == "__main__":
    unittest.main()
