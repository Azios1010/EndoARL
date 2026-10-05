"""PPO orchestration for a frozen validated hybrid; --dry-run never trains.

SERVER_VALIDATION_REQUIRED: validated supervised hybrid checkpoint, verified
GT videos and measured cost calibration must exist locally. Construction and
configuration checks perform no installs/downloads/model-hub requests.
"""
import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid_memory.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--hybrid-checkpoint", help="Validated supervised hybrid checkpoint")
    parser.add_argument('--legacy-baseline-report', help='Verified held-out LEGACY_OFFICIAL report for the separate legacy + PPO ablation')
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--validate-modules", action="store_true", help="Construct only tiny actor/critic, no reconstruction model")
    parser.add_argument("--reference", action="store_true", help="Explicit CPU reference configuration, no efficiency claims")
    parser.add_argument("--trusted-checkpoints", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--output", default="checkpoints/ppo_write_training.pt")
    parser.add_argument("--actor-output", default="checkpoints/ppo_write_actor.pt")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from dust3r.memory.config import load_config
    from dust3r.training.hybrid_trainer import primitive_config, validate_required_paths
    config = load_config(args.config)
    raw = primitive_config(config)
    if args.reference:
        raw["memory"]["short"]["backend"] = "math_reference"
        raw["memory"]["short"]["require_flash"] = False
        raw["memory"]["long"]["backend"] = "torch_reference"
        raw["uncertainty"]["enabled"] = False
    if args.manifest:
        raw["training"]["data_manifest"] = args.manifest
    config = load_config(raw)
    requirements = validate_required_paths(config)
    legacy = config.memory.backend == 'legacy'
    if legacy:
        if config.baseline_identity != 'LEGACY_POLICY':
            raise ValueError('Legacy PPO ablation must be explicitly labelled LEGACY_POLICY')
        if args.hybrid_checkpoint:
            raise ValueError('Legacy PPO does not load hybrid weights')
        if not args.legacy_baseline_report or not Path(args.legacy_baseline_report).is_file():
            requirements.append('Legacy PPO requires --legacy-baseline-report from held-out LEGACY_OFFICIAL evaluation')
        if not Path('checkpoints/raft-things.pth').is_file():
            requirements.append('Legacy memory retains mandatory checkpoints/raft-things.pth even with uncertainty disabled')
    elif not args.hybrid_checkpoint or not Path(args.hybrid_checkpoint).is_file():
        requirements.append(f"Validated local supervised hybrid checkpoint required: {args.hybrid_checkpoint!r}")
    if config.rl.reward_raft_weight or config.rl.reward_kda_geometry_weight:
        if not config.rl.cost_calibration_path or not Path(config.rl.cost_calibration_path).is_file():
            requirements.append("Compute-sensitive PPO requires measured local rl.cost_calibration_path")
    report = {"dry_run": args.dry_run, "config_valid": True, "actor_state_dim": config.rl.actor_state_dim,
              "actor_hidden": config.rl.actor_hidden, "critic_hidden": config.rl.critic_hidden,
              "frozen_environment": config.rl.frozen_environment,
              "requirements_ready": not requirements, "blocking_requirements": requirements,
              "actions": ["TEMP_ONLY", "WRITE_LONG"], "no_downloads": True, "no_installations": True,
              "server_validation": "SERVER_VALIDATION_REQUIRED: stable supervised checkpoint and calibrated delayed reward"}
    if args.validate_data:
        from dust3r.training.sequence_dataset import SequenceDataset
        report["dataset"] = SequenceDataset(config.training.data_manifest).validate_all()
    if args.validate_modules:
        from dust3r.rl.networks import Actor, Critic
        actor = Actor(hidden=tuple(config.rl.actor_hidden))
        critic = Critic(34 if config.rl.privileged_critic else 30, tuple(config.rl.critic_hidden))
        report["actor_parameters"] = sum(parameter.numel() for parameter in actor.parameters())
        report["critic_parameters"] = sum(parameter.numel() for parameter in critic.parameters())
        report["separate_actor_critic"] = not ({id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()})
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    if requirements:
        raise FileNotFoundError("PPO readiness failed:\n" + "\n".join(requirements))
    if args.updates < 1:
        raise ValueError("--updates must be positive")
    if config.rl.reward_mode != "dense_gt":
        raise ValueError("This v1 entrypoint selects verified DenseGTReward. Other reward interfaces require an explicit data protocol; no fabricated dense GT fallback.")
    import torch
    from dust3r.training.hybrid_trainer import (load_initialized_model, load_training_checkpoint,
                                               configure_trainable_parameters, move_tensors,
                                               runtime_revisions, SOURCE_COMMIT)
    from dust3r.training.sequence_dataset import SequenceDataset
    from dust3r.training.teacher import file_sha256
    from dust3r.training.losses import pointmap_from_output
    from dust3r.memory.policies import PPOWritePolicy
    from dust3r.rl.networks import Actor, Critic, privileged_critic_observation
    from dust3r.rl.normalization import ObservationNormalizer
    from dust3r.rl.environment import Environment
    from dust3r.rl.rewards import DenseGTReward, CostCalibration
    from dust3r.rl.rollout import RolloutBuffer, Transition
    from dust3r.rl.ppo import PPO, PPOConfig
    initialization_config = primitive_config(config)
    initialization_config["policy"]["long_write_policy"] = 'always' if legacy else 'heuristic'
    model, hashes = load_initialized_model(load_config(initialization_config), args.device, args.trusted_checkpoints)
    if legacy:
        from dust3r.training.validation import validate_legacy_baseline_report
        baseline_report = json.loads(Path(args.legacy_baseline_report).read_text(encoding='utf-8'))
        validate_legacy_baseline_report(baseline_report, hashes['endo3r'],
                                        file_sha256(config.training.data_manifest))
    else:
        checkpoint = load_training_checkpoint(args.hybrid_checkpoint, model)
        if checkpoint.get("validation_summary", {}).get("validated_for_ppo") is not True:
            raise ValueError("Hybrid checkpoint has no successful held-out validated_for_ppo gate; PPO remains blocked")
    configure_trainable_parameters(model, "r0")
    model.eval()
    actor = Actor(hidden=tuple(config.rl.actor_hidden)).to(args.device)
    critic = Critic(34 if config.rl.privileged_critic else 30, tuple(config.rl.critic_hidden)).to(args.device)
    normalizer = ObservationNormalizer().to(args.device)
    policy = PPOWritePolicy(actor, normalizer=normalizer)
    runner = model.make_online_runner(training=False, policy=policy)
    calibration = None
    if config.rl.cost_calibration_path:
        with open(config.rl.cost_calibration_path, encoding="utf-8") as handle:
            calibration = CostCalibration(**json.load(handle))
    reward = DenseGTReward(quality_weight=config.rl.reward_quality_weight, write_weight=config.rl.reward_write_weight,
                           raft_weight=config.rl.reward_raft_weight, kda_geometry_weight=config.rl.reward_kda_geometry_weight,
                           error_cap=config.rl.reward_quality_error_cap, calibration=calibration)
    dataset = SequenceDataset(config.training.data_manifest, split="train",
                              image_hw=(config.training.image_height, config.training.image_width))
    eligible = [index for index, (_, entries, _) in enumerate(dataset.samples) if len(entries) >= 4]
    if not eligible:
        raise ValueError("PPO actor training needs videos of at least four frames, including a nonforced decision")
    current_targets = []
    if legacy:
        import hashlib
        environment_id = hashlib.sha256(json.dumps({'released': hashes, 'config': primitive_config(config)},
                                                  sort_keys=True).encode()).hexdigest()
    else:
        environment_id = file_sha256(args.hybrid_checkpoint)
    environment = Environment(runner, reward, gt_provider=lambda index: current_targets[index],
                              environment_checkpoint_id=environment_id)
    optimizer = PPO(actor, critic, PPOConfig(learning_rate=config.rl.actor_lr,
                    gamma=config.rl.gamma, gae_lambda=config.rl.gae_lambda, clip_ratio=config.rl.clip_epsilon,
                    entropy_coefficient=config.rl.entropy_coef, value_coefficient=config.rl.value_coef,
                    value_clip=config.rl.value_clip, max_grad_norm=config.rl.max_grad_norm,
                    minibatch_size=config.rl.minibatch_size, epochs=config.rl.ppo_epochs,
                    target_kl=config.rl.target_kl, normalize_advantages=config.rl.advantage_normalization))
    for group in optimizer.critic_optimizer.param_groups:
        group["lr"] = config.rl.critic_lr
    packet = None
    sequence_cursor = 0

    def critic_input(observation):
        if not config.rl.privileged_critic:
            return observation
        pair = environment.current_pair
        target = current_targets[pair.candidate_index]
        pred = pointmap_from_output(pair.res1).detach().float()
        mask = target["valid_mask"]
        error = ((reward.scale * pred - target["pointmap"]).abs().mean(-1) / reward.radius_scale)[mask]
        if not torch.isfinite(error).all() or not error.numel():
            raise ValueError("Privileged critic requires valid current-frame GT error, never future GT")
        extra = torch.stack((error.mean().clamp(0, 5), torch.quantile(error, .9).clamp(0, 5),
                             mask.float().mean(), error.new_ones(())))[None]
        return privileged_critic_observation(observation, extra)

    for update in range(args.updates):
        normalization_version = normalizer.freeze()
        buffer = RolloutBuffer(config.rl.rollout_transitions)
        while len(buffer) < buffer.capacity:
            if packet is None:
                sample = dataset[eligible[sequence_cursor % len(eligible)]]
                sequence_cursor += 1
                current_targets[:] = [move_tensors(target, args.device) for target in sample.targets]
                packet = environment.reset([move_tensors(frame, args.device) for frame in sample.frames], sample.episode_id)
            raw_observation = packet.observation.detach()
            with torch.no_grad():
                observation = normalizer(raw_observation)
                critic_observation = critic_input(observation)
                value = critic(critic_observation).item()
                decision = policy.act(packet, deterministic=False)
            next_packet, scalar_reward, done, info = environment.step(decision.action)
            flags = info["write_diagnostics"] or {}
            def flag(name, default=False):
                return flags.get(name, default) if isinstance(flags, dict) else getattr(flags, name, default)
            buffer.add(Transition(raw_observation=raw_observation, observation=observation,
                       critic_observation=critic_observation, action=int(decision.action.item()),
                       log_prob=float(decision.log_prob.item()), value=value, reward=scalar_reward,
                       done=done, action_mask=decision.action_mask, forced=bool(decision.forced.item()),
                       requested_write=bool(flag("requested_write", flag("write_attempt"))),
                       successful_write=bool(flag("successful_write", flag("update_success"))),
                       raft_flag=bool(flag("raft_called")), kda_flag=bool(flag("kda_updated")),
                       quality_metrics=info["quality_metrics"], cost_components=info["cost_components"],
                       episode_id=info["episode_id"], candidate_index=info["candidate_index"],
                       arrival_index=info["arrival_index"], normalization_version=normalization_version,
                       environment_checkpoint_id=environment_id))
            packet = None if done else next_packet
        with torch.no_grad():
            last_value = 0.0 if packet is None else critic(critic_input(normalizer(packet.observation))).item()
        if not buffer.transitions[-1].done:
            buffer.transitions[-1].truncated = True
            buffer.transitions[-1].next_value = last_value
        metrics = optimizer.update(buffer, last_value)
        normalizer.unfreeze()
        normalizer.update(torch.stack([transition.raw_observation for transition in buffer.transitions]))
        print(json.dumps({"update": update, "metrics": metrics}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    training_payload = {"schema": "endo3r_ppo_write_v1", "config": primitive_config(config),
                        'source_commit': SOURCE_COMMIT, 'external_revisions': runtime_revisions(),
                        "ppo": optimizer.state_dict(), "normalizer": normalizer.state_dict(),
                        "environment_checkpoint_id": environment_id, "original_checkpoint_hashes": hashes,
                        "observation_schema": "actor30_v1", "reward_calibration": None if calibration is None else vars(calibration),
                        "torch_rng": torch.get_rng_state(), "updates": args.updates,
                        "validated_for_deployment": False}
    torch.save(training_payload, output)
    actor_output = Path(args.actor_output)
    actor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value for key, value in training_payload.items() if key != "ppo"} |
               {"actor": actor.state_dict()}, actor_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
