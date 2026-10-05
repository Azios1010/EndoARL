"""Strict, dependency-light configuration for hybrid memory and PPO."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Union


FLA_REVISION = "9f38d24980c46d46bd38614e743cdacd21906578"


@dataclass
class CheckpointConfig:
    dust3r: Optional[str] = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    endo3r: Optional[str] = "checkpoints/endo3r.pth"
    raft: Optional[str] = "checkpoints/raft-things.pth"


@dataclass
class ShortMemoryConfig:
    max_frames: int = 4
    model_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    backend: str = "flash"
    require_flash: bool = True
    dropout: float = 0.0
    store_projected: bool = True
    raw_shadow_training_only: bool = True


@dataclass
class LongMemoryConfig:
    backend: str = "fla_kda"
    implementation_revision: str = FLA_REVISION
    model_dim: int = 256
    num_heads: int = 4
    key_dim: int = 64
    value_dim: int = 64
    state_dtype: str = "float32"
    state_v_first: bool = False
    chunk_by_frame: bool = True
    kernel_chunk_size: int = 64
    inference_mode: str = "chunk"
    qk_l2_normalize: bool = True
    uncertainty_write_gate: bool = True
    confidence_gates_beta_and_decay: bool = True
    decay_normalization: str = "divide_by_frame_patches"
    initial_beta: float = 0.1
    initial_frame_decay_rate: float = 0.1


@dataclass
class FusionConfig:
    type: str = "channel_gate"
    hidden_dim: int = 64
    output_dim: int = 1024
    initial_short_preference: float = 0.75


@dataclass
class MemoryConfig:
    backend: str = "hybrid_flash_kda"
    input_dim: int = 1024
    output_dim: int = 1024
    short: ShortMemoryConfig = field(default_factory=ShortMemoryConfig)
    long: LongMemoryConfig = field(default_factory=LongMemoryConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class UncertaintyConfig:
    enabled: bool = True
    estimator: str = "raft"
    run_on: str = "write_long_only"
    reference: str = "last_successful_long_write"
    raft_iters: int = 20
    raft_dtype: str = "float32"
    raft_mixed_precision: bool = False
    distance_scale_fraction: float = 0.1
    confidence_alpha: float = 5.0
    confidence_floor_on_valid: float = 0.05
    invalid_as_high_uncertainty: bool = True
    reject_all_invalid_write: bool = True


@dataclass
class PolicyConfig:
    long_write_policy: str = "heuristic"
    actions: list = field(default_factory=lambda: ["temp_only", "write_long"])
    force_first_write: bool = True
    terminal_pair_action: str = "none"
    observation_schema: str = "actor30_v1"
    heuristic_similarity: float = 0.92
    heuristic_recent_long_descriptors: int = 4
    deterministic_eval: bool = True


@dataclass
class TrainingConfig:
    entrypoint: str = "train_hybrid_memory.py"
    data_manifest: Optional[str] = None
    image_width: int = 320
    image_height: int = 256
    batch_size: int = 1
    sequence_raw_frames: int = 9
    tbptt_pairs: int = 4
    stride: int = 1
    new_memory_dtype: str = "bfloat16"
    pretrained_compute_dtype: str = "float32"
    freeze_image_encoder: bool = True
    freeze_decoder_t1: bool = True
    freeze_heads_t1: bool = True
    freeze_key_value_t1: bool = True
    freeze_shared_norms_t1: bool = True
    optimizer: str = "adamw"
    lr_new: float = 1e-4
    lr_key_value_norm_ft: float = 1e-5
    lr_last_decoder_heads_ft: float = 5e-6
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    objective: str = "masked_common_frame_pointmap_l1"
    scale_alignment: str = "first_frame_detached_episode_scale"
    loss_weights: dict = field(default_factory=lambda: dict(pointmap=1.0, depth=0.0,
                                                           confidence=0.0, pose=0.0, temporal=0.0))
    teacher_distillation: str = "optional_warm_start"
    teacher_weight_initial: float = 0.1
    teacher_cosine_weight: float = 0.1
    teacher_anneal_fraction: float = 0.1


@dataclass
class RLConfig:
    entrypoint: str = "train_rl_controller.py"
    algorithm: str = "ppo"
    actor_state_dim: int = 30
    actor_hidden: list = field(default_factory=lambda: [64, 64])
    critic_hidden: list = field(default_factory=lambda: [64, 64])
    privileged_critic: bool = False
    frozen_environment: bool = True
    reward_mode: str = "dense_gt"
    reward_quality_weight: float = 1.0
    reward_quality_error_cap: float = 5.0
    failure_quality_error: float = 5.0
    reward_write_weight: float = 0.0
    reward_raft_weight: float = 0.01
    reward_kda_geometry_weight: float = 0.01
    terminal_quality_weight: float = 0.0
    cost_calibration_path: Optional[str] = None
    rollout_transitions: int = 128
    minibatch_size: int = 64
    ppo_epochs: int = 4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    value_clip: float = 0.2
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    advantage_normalization: bool = True
    reward_normalization: str = "fixed_gt_scale_and_cost_ratios"
    observation_normalizer: str = "continuous_indices_only"


@dataclass
class EvaluationConfig:
    include_terminal_write_in_rates: bool = False
    export_state_between_patients: bool = False
    quality_relative_tolerance: float = 0.05
    diagnostics: str = "scalar_only"


@dataclass
class HybridConfig:
    schema_version: int = 1
    baseline_identity: str = "NEW_HYBRID"
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "HybridConfig":
        _validate_finite(asdict(self))
        if self.schema_version != 1:
            raise ValueError("Unsupported hybrid schema_version; expected 1")
        if self.baseline_identity not in {"NEW_HYBRID", "LEGACY_OFFICIAL", "LEGACY_POLICY", "LEGACY_CLEAN"}:
            raise ValueError("Unknown baseline_identity")
        validate_memory_config(self.memory)
        if self.policy.actions != ["temp_only", "write_long"] or not self.policy.force_first_write:
            raise ValueError("v1 requires exactly TEMP_ONLY/WRITE_LONG and a forced first write")
        if self.policy.long_write_policy not in {"heuristic", "always", "never", "ppo"}:
            raise ValueError("Unknown long_write_policy")
        if self.policy.observation_schema != "actor30_v1" or self.policy.terminal_pair_action != "none":
            raise ValueError("v1 requires actor30_v1 and no terminal-pair action")
        if not -1 <= self.policy.heuristic_similarity <= 1:
            raise ValueError("heuristic_similarity must be in [-1,1]")
        if not 1 <= self.policy.heuristic_recent_long_descriptors <= 4:
            raise ValueError("v1 heuristic retains 1..4 descriptors")
        u = self.uncertainty
        if (u.estimator != "raft" or u.run_on != "write_long_only" or
                u.reference != "last_successful_long_write" or u.raft_dtype != "float32" or
                u.raft_mixed_precision or not u.invalid_as_high_uncertainty or not u.reject_all_invalid_write):
            raise ValueError("Uncertainty must use safe FP32 WRITE-only RAFT semantics")
        if u.raft_iters <= 0 or u.distance_scale_fraction <= 0 or u.confidence_alpha < 0:
            raise ValueError("Invalid uncertainty iteration/scale/confidence values")
        if not 0 <= u.confidence_floor_on_valid <= 1:
            raise ValueError("confidence_floor_on_valid must be in [0,1]")
        t, r = self.training, self.rl
        if t.batch_size != 1 or min(t.image_height, t.image_width, t.tbptt_pairs, t.stride) <= 0:
            raise ValueError("v1 supports batch_size=1 and positive dimensions/TBPTT/stride")
        if t.sequence_raw_frames < 2 or t.tbptt_pairs > t.sequence_raw_frames - 1:
            raise ValueError("TBPTT length cannot exceed available decoded pairs")
        if t.optimizer != "adamw" or t.objective != "masked_common_frame_pointmap_l1":
            raise ValueError("Unsupported v1 training optimizer/objective")
        if t.new_memory_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported memory activation dtype")
        if t.pretrained_compute_dtype != 'float32':
            raise ValueError('v1 pretrained encoder/decoder compute remains float32')
        if min(t.lr_new, t.lr_key_value_norm_ft, t.lr_last_decoder_heads_ft, t.gradient_clip) <= 0 or t.weight_decay < 0:
            raise ValueError('Training learning rates/clipping must be positive and weight decay nonnegative')
        if set(t.loss_weights) != {"pointmap", "depth", "confidence", "pose", "temporal"}:
            raise ValueError("loss_weights must explicitly name pointmap/depth/confidence/pose/temporal")
        if any(value < 0 for value in t.loss_weights.values()):
            raise ValueError("loss weights cannot be negative")
        if r.algorithm != "ppo" or r.actor_state_dim != 30 or not r.frozen_environment:
            raise ValueError("v1 PPO requires actor30 and a frozen reconstruction environment")
        if r.reward_mode not in {"dense_gt", "sequence_gt", "teacher_proxy"}:
            raise ValueError("Unknown reward_mode")
        if not 0 <= r.gamma <= 1 or not 0 <= r.gae_lambda <= 1:
            raise ValueError("PPO discounts must be in [0,1]")
        if min(r.rollout_transitions, r.minibatch_size, r.ppo_epochs, r.actor_lr, r.critic_lr) <= 0:
            raise ValueError("PPO batch sizes/epochs/learning rates must be positive")
        if min(r.clip_epsilon, r.max_grad_norm, r.target_kl, r.reward_quality_error_cap, r.failure_quality_error) <= 0:
            raise ValueError('PPO clipping/KL/quality bounds must be positive')
        if min(r.entropy_coef, r.value_coef, r.value_clip, r.reward_quality_weight,
               r.reward_write_weight, r.reward_raft_weight, r.reward_kda_geometry_weight,
               r.terminal_quality_weight) < 0:
            raise ValueError('PPO loss/reward weights cannot be negative')
        for hidden in (r.actor_hidden, r.critic_hidden):
            if not hidden or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in hidden):
                raise ValueError('Actor/critic hidden dimensions must be positive integers')
        if self.evaluation.export_state_between_patients:
            raise ValueError("Memory state must reset between patients")
        return self


def _validate_finite(value, path='config'):
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite(child, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite(child, f'{path}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{path} must be finite')


def validate_memory_config(config: MemoryConfig) -> MemoryConfig:
    _validate_finite(asdict(config), 'memory')
    if config.backend not in {"legacy", "hybrid_flash_kda"}:
        raise ValueError("memory.backend must be legacy or hybrid_flash_kda")
    if config.input_dim <= 0 or config.output_dim != config.input_dim:
        raise ValueError("Memory residual requires equal positive input/output dimensions")
    s, l, f = config.short, config.long, config.fusion
    if s.model_dim != s.num_heads * s.head_dim or min(s.num_heads, s.head_dim) <= 0:
        raise ValueError("short.model_dim must equal num_heads*head_dim")
    if not 1 <= s.max_frames <= 4 or s.dropout != 0 or not s.store_projected:
        raise ValueError("v1 short memory requires 1..4 projected frames and zero dropout")
    if s.backend not in {"flash", "sdpa", "sdpa_flash", "flash_attn", "math_reference"}:
        raise ValueError("Unsupported exact short attention backend")
    if s.require_flash and s.backend in {"math_reference", "sdpa"}:
        raise ValueError("require_flash cannot claim generic SDPA/math as verified Flash")
    if l.backend not in {"fla_kda", "torch_reference"}:
        raise ValueError("long.backend must be fla_kda or torch_reference")
    if min(l.num_heads, l.key_dim, l.value_dim) <= 0 or l.model_dim != l.num_heads * l.key_dim:
        raise ValueError("long.model_dim must equal num_heads*key_dim")
    if (l.state_dtype != "float32" or l.state_v_first or not l.chunk_by_frame or
            not l.qk_l2_normalize or not l.confidence_gates_beta_and_decay or
            l.decay_normalization != "divide_by_frame_patches"):
        raise ValueError("v1 KDA requires FP32 key-first frame state and confidence-gated beta/decay")
    if l.kernel_chunk_size not in {32, 64} or l.inference_mode not in {"chunk", "fused_recurrent"}:
        raise ValueError("Unsupported KDA chunk/inference mode")
    if not 0 < l.initial_beta < 1 or l.initial_frame_decay_rate <= 0:
        raise ValueError("Initial KDA gates must be finite positive rates")
    if l.backend == "fla_kda" and (l.key_dim > 256 or l.implementation_revision != FLA_REVISION):
        raise ValueError("FLA adapter requires the pinned revision and key_dim<=256")
    if f.type != "channel_gate" or f.output_dim != config.output_dim or f.hidden_dim <= 0:
        raise ValueError("Invalid channel fusion dimensions/type")
    if not 0 < f.initial_short_preference < 1:
        raise ValueError("Fusion initial preference must be in (0,1)")
    return config


def _update_dataclass(instance: Any, values: Mapping, path: str = "config") -> Any:
    if not isinstance(values, Mapping):
        raise ValueError(f"{path} must be a mapping")
    allowed = {item.name for item in fields(instance)}
    extra = set(values) - allowed
    if extra:
        raise ValueError(f"Unknown {path} fields: {', '.join(sorted(extra))}")
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            value = _update_dataclass(current, value, f"{path}.{name}")
        elif isinstance(current, bool) and not isinstance(value, bool):
            raise ValueError(f"{path}.{name} must be a boolean")
        elif isinstance(current, int) and not isinstance(current, bool) and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{path}.{name} must be an integer")
        elif isinstance(current, float) and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError(f"{path}.{name} must be numeric")
        elif isinstance(current, str) and not isinstance(value, str):
            raise ValueError(f"{path}.{name} must be a string")
        elif isinstance(current, list) and not isinstance(value, list):
            raise ValueError(f"{path}.{name} must be a list")
        elif isinstance(current, dict) and not isinstance(value, Mapping):
            raise ValueError(f"{path}.{name} must be a mapping")
        elif current is None and value is not None and not isinstance(value, str):
            raise ValueError(f"{path}.{name} must be a path string or null")
        setattr(instance, name, value)
    return instance


def load_config(source: Union[str, Path, Mapping, HybridConfig, None] = None) -> HybridConfig:
    if isinstance(source, HybridConfig):
        return source.validate()
    if source is None:
        return HybridConfig().validate()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Hybrid configuration not found: {path}")
        if path.suffix.lower() == ".json":
            source = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML parsing requires PyYAML; use a JSON file or mapping without it") from exc
            source = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _update_dataclass(HybridConfig(), source).validate()
