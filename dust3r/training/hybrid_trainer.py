"""Explicit supervised training driver, optimizer groups and bounded TBPTT.

No code here invokes the broken legacy Endo3R.forward training branch. This
module imports only standard-library/PyTorch components until model loading is
explicitly requested. Running the supplied dry CLI never starts optimization.
"""
from dataclasses import asdict, dataclass, is_dataclass
import math
from pathlib import Path
import random
from typing import Mapping

import torch
from torch import nn

from .losses import EpisodeScale, ReconstructionLoss, feature_distillation_loss, pointmap_from_output
from .teacher import file_sha256, warm_start_weight


CHECKPOINT_SCHEMA = "endo3r_hybrid_training_v1"
SOURCE_COMMIT = "8145c8e5535d2111974561e7d1cc06f1d008fe5e"
FLA_REVISION = "9f38d24980c46d46bd38614e743cdacd21906578"


def runtime_revisions():
    from importlib.metadata import PackageNotFoundError, version
    result = {'fla_expected_revision': FLA_REVISION, 'torch': str(torch.__version__),
              'torch_cuda_build': torch.version.cuda}
    for name in ('flash-attn', 'flash-linear-attention', 'triton'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def config_value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def primitive_config(config):
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return {key: primitive_config(value) for key, value in config.items()}
    if isinstance(config, (tuple, list)):
        return [primitive_config(value) for value in config]
    return config


def _memory_prefix(name):
    return name.startswith(("hybrid_memory.", "memory_module.", "memory.short.", "memory.long.", "memory.fusion."))


def configure_trainable_parameters(model: nn.Module, stage="t1", unfreeze_decoder=False, unfreeze_heads=False):
    if stage not in {"t1", "t2", "r0", "r1_supervised"}:
        raise ValueError("Stage must be t1, t2, r0, or r1_supervised")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    value_prefixes = ("attn_head_1.", "attn_head_2.", "value_encoder.", "value_norm.",
                      "value_out.", "pos_patch_embed.", "norm_q.", "norm_k.", "norm_v.")
    last_decoder_prefixes = []
    if stage == "t2" and unfreeze_decoder:
        for branch in ("dec_blocks", "dec_blocks2"):
            blocks = getattr(getattr(model, "dust3r", None), branch, ())
            last_decoder_prefixes.extend(f"dust3r.{branch}.{index}." for index in range(max(0, len(blocks) - 2), len(blocks)))
    for name, parameter in model.named_parameters():
        if _memory_prefix(name) and ".norms." not in name and not any(word in name.lower() for word in ("raft", "uncertainty", "actor", "critic")):
            if stage in {"t1", "t2"} or stage == "r1_supervised" and any(part in name for part in (".fusion.", ".long.")):
                parameter.requires_grad_(True)
        if stage == "t2" and name.startswith(value_prefixes):
            parameter.requires_grad_(True)
        if stage == "t2" and _memory_prefix(name) and ".norms." in name:
            parameter.requires_grad_(True)
        if stage == "t2" and last_decoder_prefixes and name.startswith(tuple(last_decoder_prefixes)):
            parameter.requires_grad_(True)
        if stage == "t2" and unfreeze_heads and name.startswith(("dust3r.downstream_head", "dust3r.head")):
            parameter.requires_grad_(True)
    # frozen RAFT must remain eval even when reconstruction is in train mode
    for name, module in model.named_modules():
        if "raft" in name.lower():
            module.eval()
    return {"trainable": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "frozen": [name for name, parameter in model.named_parameters() if not parameter.requires_grad]}


def build_optimizer_groups(model, training_config=None, stage="t1"):
    lr_new = float(config_value(training_config, "lr_new", 1e-4))
    lr_value = float(config_value(training_config, "lr_key_value_norm_ft", 1e-5))
    lr_decoder = float(config_value(training_config, "lr_last_decoder_heads_ft", 5e-6))
    weight_decay = float(config_value(training_config, "weight_decay", 0.01))
    grouped = {}
    parameter_ids = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in parameter_ids:
            raise ValueError("A shared parameter was registered into two optimizer groups")
        parameter_ids.add(id(parameter))
        if _memory_prefix(name) and ".norms." in name:
            label, lr = "key_value_norm", lr_value
        elif _memory_prefix(name):
            label, lr = "memory", 1e-6 if stage == "r1_supervised" else lr_new
        elif name.startswith("dust3r."):
            label, lr = "decoder_heads", lr_decoder
        else:
            label, lr = "key_value_norm", lr_value
        no_decay = parameter.ndim < 2 or name.endswith(".bias") or any(part in name.lower() for part in ("norm", "a_log", "dt_bias"))
        key = (label, no_decay)
        group = grouped.setdefault(key, {"params": [], "names": [], "lr": lr,
                                         "weight_decay": 0.0 if no_decay else weight_decay,
                                         "group_name": label + ("_no_decay" if no_decay else "_decay")})
        group["params"].append(parameter)
        group["names"].append(name)
    if not grouped:
        raise ValueError("No trainable parameters: verify new memory registration and training stage")
    return list(grouped.values())


def optimizer_group_summary(groups):
    return [{"name": group["group_name"], "lr": group["lr"], "weight_decay": group["weight_decay"],
             "parameter_tensors": len(group["params"]), "parameters": sum(p.numel() for p in group["params"]),
             "names": group["names"]} for group in groups]


def make_optimizer(model, training_config=None, stage="t1"):
    groups = build_optimizer_groups(model, training_config, stage)
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def make_lr_scheduler(optimizer, total_steps, warmup_fraction=0.05, min_lr=1e-6):
    if total_steps <= 0:
        raise ValueError("A positive explicit optimization step budget is required")
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    functions = []
    for group in optimizer.param_groups:
        floor = min(1.0, min_lr / group["lr"])
        def schedule(step, floor=floor):
            if step < warmup_steps:
                return max(1e-3, (step + 1) / warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        functions.append(schedule)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)


def move_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_tensors(item, device) for item in value)
    return value


def _pair_value(pair, name):
    return pair[name] if isinstance(pair, Mapping) else getattr(pair, name)


class HybridTrainer:
    def __init__(self, model, runner, config, optimizer=None, loss=None, stage="t1",
                 total_steps=1000, teacher_cache_factory=None):
        self.model, self.runner, self.config = model, runner, config
        self.training_config = config_value(config, "training", config)
        self.stage = stage
        self.tbptt_pairs = int(config_value(self.training_config, "tbptt_pairs", 4))
        if self.tbptt_pairs < 1:
            raise ValueError("tbptt_pairs must be positive")
        self.optimizer = optimizer or make_optimizer(model, self.training_config, stage)
        self.scheduler = make_lr_scheduler(self.optimizer, total_steps)
        weights = config_value(self.training_config, "loss_weights", {})
        if any(float(config_value(weights, key, 0)) != 0 for key in ("depth", "pose", "temporal")):
            raise ValueError("v1 has no supported depth/pose/temporal training loss; select pointmap supervision explicitly")
        self.loss = loss or ReconstructionLoss(float(config_value(weights, "pointmap", 1.0)),
                                               float(config_value(weights, "confidence", 0.0)))
        self.total_steps = total_steps
        self.global_step = 0
        self.teacher_cache_factory = teacher_cache_factory
        self.device = next(model.parameters()).device
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda' and
            config_value(self.training_config, 'new_memory_dtype') == 'float16'))
        self.deterministic_policy = getattr(runner, "policy", None)

    def _optimize_window(self, losses):
        combined = torch.stack(losses).mean()
        if not torch.isfinite(combined):
            raise FloatingPointError("Nonfinite supervised loss; optimizer step rejected")
        self.optimizer.zero_grad(set_to_none=True)
        # The first pair may have no memory input and no trainable graph.
        if combined.requires_grad:
            self.scaler.scale(combined).backward()
            self.scaler.unscale_(self.optimizer)
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad and parameter.grad is not None]
            if not parameters:
                raise RuntimeError("Supervised graph contains no new-module gradients")
            if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise FloatingPointError("Nonfinite training gradients; optimizer step rejected")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(config_value(self.training_config, "gradient_clip", 1.0)))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
        else:
            gradient_norm = combined.new_zeros(())
        # Reprojection happens AFTER the update, after consuming the whole graph.
        self.runner.detach_state(refresh_short=True)
        return {"loss": float(combined.detach()), "gradient_norm": float(gradient_norm), "step": self.global_step}

    def train_sequence(self, sample):
        if len(sample.frames) < 3:
            raise ValueError("Attention training requires at least three frames for a memory-dependent read")
        self.model.train()
        self.runner.reset(episode_id=sample.episode_id, training=True)
        targets = [move_tensors(target, self.device) for target in sample.targets]
        window, logs, scale = [], [], None
        cache = self.teacher_cache_factory(sample) if self.teacher_cache_factory is not None else None
        for index, frame in enumerate(sample.frames):
            if self.stage == "t1" and self.deterministic_policy is not None:
                from dust3r.memory.policies import AlwaysWritePolicy
                # Explicit deterministic curriculum, no actor/PPO execution.
                self.runner.policy = (AlwaysWritePolicy() if self.global_step < max(1, round(self.total_steps * .1))
                                      else self.deterministic_policy)
            pair = self.runner.step(move_tensors(frame, self.device), terminal=index == len(sample.frames) - 1)
            if pair is None:
                continue
            candidate_index = int(_pair_value(pair, "candidate_index"))
            res1, res2 = _pair_value(pair, "res1"), _pair_value(pair, "res2")
            if scale is None:
                scale = EpisodeScale.fit(pointmap_from_output(res1), targets[0]["pointmap"], targets[0]["valid_mask"])
            current_loss, metrics = self.loss(res1, res2, targets[candidate_index], targets[candidate_index + 1], scale)
            if cache is not None and candidate_index > 0:
                weight = warm_start_weight(self.global_step, self.total_steps,
                                           float(config_value(self.training_config, "teacher_weight_initial", 0.1)),
                                           float(config_value(self.training_config, "teacher_anneal_fraction", 0.1)))
                if weight:
                    feature = _pair_value(pair, "memory_feature")
                    teacher = cache.load(candidate_index, int(_pair_value(pair, "arrival_index")), self.device)
                    current_loss = current_loss + weight * feature_distillation_loss(
                        feature, teacher, float(config_value(self.training_config, "teacher_cosine_weight", 0.1)))
            window.append(current_loss)
            if len(window) == self.tbptt_pairs or index == len(sample.frames) - 1:
                logs.append(self._optimize_window(window))
                window.clear()
            if self.global_step >= self.total_steps:
                break
        return logs

    def save_checkpoint(self, path, validation_summary=None, original_checkpoint_hashes=None):
        save_training_checkpoint(path, self.model, self.config, self.optimizer, self.scheduler,
                                 self.global_step, self.stage, validation_summary, original_checkpoint_hashes,
                                 self.scaler)


def save_training_checkpoint(path, model, config, optimizer=None, scheduler=None, global_step=0,
                             stage="t1", validation_summary=None, original_checkpoint_hashes=None,
                             scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.state_dict()
    # Episode state is ordinary runner-owned data, never registered model buffers.
    forbidden = ("last_long_reference", "short_raw_", "kda_state", "mem_attn", "mem_count")
    if any(any(word in key for word in forbidden) for key in model_state):
        raise ValueError("Patient/episodic memory must not be saved as model weights")
    from .provenance import capture_rng_state, source_provenance
    provenance = source_provenance()
    payload = {"schema": CHECKPOINT_SCHEMA, "schema_version": 1, "source_commit": provenance['git_commit'],
               "audited_legacy_commit": SOURCE_COMMIT, "source_provenance": provenance,
               "config": primitive_config(config), "model": model_state, "stage": stage,
               "global_step": global_step, "original_checkpoint_hashes": original_checkpoint_hashes or {},
               "external_revisions": runtime_revisions(), "observation_schema": "actor30_v1",
               "validation_summary": validation_summary or {}, "rng": capture_rng_state(),
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "scheduler": None if scheduler is None else scheduler.state_dict(),
               "scaler": None if scaler is None else scaler.state_dict()}
    torch.save(payload, path)


def load_training_checkpoint(path, model, expected_backend="hybrid_flash_kda", optimizer=None,
                             scheduler=None, restore_rng=False, scaler=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local hybrid checkpoint required: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint is not an explicit hybrid training checkpoint")
    backend = payload.get("config", {}).get("memory", {}).get("backend")
    if backend != expected_backend:
        raise ValueError(f"Checkpoint backend {backend!r} cannot load into {expected_backend!r}")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get('scaler') is not None:
        scaler.load_state_dict(payload['scaler'])
    if restore_rng:
        from .provenance import restore_rng_state
        restore_rng_state(payload['rng'])
    return payload


def validate_required_paths(config, manifest=None, require_raft=None):
    checkpoints = config_value(config, "checkpoints", {})
    uncertainty = config_value(config, "uncertainty", {})
    if require_raft is None:
        require_raft = bool(config_value(uncertainty, "enabled", True))
    errors = []
    required = ["dust3r", "endo3r"] + (["raft"] if require_raft else [])
    for name in required:
        path = config_value(checkpoints, name)
        if not path or not Path(path).is_file():
            errors.append(f"Required local {name} checkpoint missing: {path!r}; no downloads are performed")
    if manifest is None:
        manifest = config_value(config_value(config, "training", {}), "data_manifest")
    if not manifest or not Path(manifest).is_file():
        errors.append(f"Verified GT sequence manifest required: {manifest!r}")
    return errors


def load_initialized_model(config, device="cpu", trusted_checkpoint=False):
    """Load ONLY existing files; explicit trusted flag covers legacy pickle args."""
    checkpoints = config_value(config, "checkpoints", {})
    for name in ("dust3r", "endo3r"):
        path = config_value(checkpoints, name)
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Required local {name} checkpoint missing: {path!r}. No model hubs/downloads are used.")
    if not trusted_checkpoint:
        raise ValueError("Released DUSt3R metadata uses pickle; pass explicit --trusted-checkpoints only for verified local files")
    try:
        from dust3r.model import Endo3R
    except ImportError as error:
        raise RuntimeError(f"Endo3R dependency unavailable: {error}. Prepare the server separately; nothing was installed.") from error
    # Scoped allowlist for the released argparse metadata, only after explicit trust.
    # The original load_model function/default legacy behavior stays unchanged.
    import argparse
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = Endo3R(dus3r_name=str(config_value(checkpoints, "dust3r")), memory_config=config).to(device)
    payload = torch.load(config_value(checkpoints, "endo3r"), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Released Endo3R checkpoint must contain a 'model' state dictionary")
    status = model.load_state_dict(payload["model"], strict=False)
    unapproved_missing = [key for key in status.missing_keys if not key.startswith(("hybrid_memory.", "hybrid_uncertainty.", "long_write_policy."))]
    if unapproved_missing or status.unexpected_keys:
        raise ValueError(f"Checkpoint mismatch beyond newly initialized hybrid modules: missing={unapproved_missing}, unexpected={status.unexpected_keys}")
    names = ["dust3r", "endo3r"]
    if config_value(config_value(config, "uncertainty", {}), "enabled", False):
        names.append("raft")
    return model, {name: file_sha256(config_value(checkpoints, name)) for name in names}
