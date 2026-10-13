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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-jsonl")
    parser.add_argument("--resume-checkpoint", help="Same-environment PPO training state, reset episodes at resume")
    parser.add_argument("--hybrid-checkpoint", help="Validated supervised hybrid checkpoint")
    parser.add_argument('--legacy-baseline-report', help='Verified held-out LEGACY_OFFICIAL report for the separate legacy + PPO ablation')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-training", action="store_true", help="Explicit authorization for server PPO optimization; default validates only")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--validate-modules", action="store_true", help="Construct only tiny actor/critic, no reconstruction model")
    parser.add_argument("--reference", action="store_true", help="Explicit CPU reference configuration, no efficiency claims")
    parser.add_argument("--trusted-checkpoints", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--output", default="checkpoints/ppo_write_training.pt")
    parser.add_argument("--actor-output", default="checkpoints/ppo_write_actor.pt")
    return parser


def load_cost_calibration(path, config, device, *, runtime=None):
    """Load identified measured proxies for this hardware/architecture/mode.

    Source checkpoint hashes remain audit metadata. Cost proxies may be reused
    across training seeds with the same architecture and uncertainty mode;
    they are not bound to one seed's reconstructed weight file.
    """
    from dataclasses import fields
    from dust3r.rl.rewards import CostCalibration
    from dust3r.experiments.benchmark import runtime_metadata
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if (payload.get('schema') != 'endo3r_cost_calibration_v1' or
            payload.get('units') != 'milliseconds' or
            not isinstance(payload.get('source_profile_sha256'), str) or
            len(payload['source_profile_sha256']) != 64):
        raise ValueError('PPO requires identified endo3r_cost_calibration_v1 measured server costs. '
                         'Regenerate with tools/calibrate_costs.py; bare/guessed cost JSON is not accepted.')
    if payload.get('timing_scope') != 'synchronized_exclusive_components_not_headline_throughput':
        raise ValueError('PPO cost calibration must come from exclusive synchronized component timings.')
    runtime = runtime if runtime is not None else runtime_metadata(device)
    if not payload.get('hardware_id') or payload['hardware_id'] != runtime.get('hardware_id'):
        raise ValueError('Cost calibration hardware/runtime differs from this server; profile this device again.')
    if payload.get('image_hw') != [config.training.image_height, config.training.image_width]:
        raise ValueError('Cost calibration resolution differs from configured image_hw.')
    if payload.get('uncertainty_metric') != config.uncertainty.metric:
        raise ValueError('Cost calibration uncertainty metric differs; Sampson and 3D geometry need separate measured costs.')
    if payload.get('uncertainty_enabled') is not config.uncertainty.enabled:
        raise ValueError('Cost calibration uncertainty enabled/disabled mode differs or is missing; reprofile this mode.')
    if payload.get('memory_config') != config.to_dict()['memory']:
        raise ValueError('Cost calibration memory architecture/backend differs or is missing; regenerate the identified profile.')
    precision = {'new_memory_dtype': config.training.new_memory_dtype,
                 'pretrained_compute_dtype': config.training.pretrained_compute_dtype}
    if payload.get('precision_config') != precision:
        raise ValueError('Cost calibration precision configuration differs or is missing; reprofile this compute mode.')
    selected = {item.name: payload[item.name] for item in fields(CostCalibration) if item.name in payload}
    if 'image_hw' in selected:
        selected['image_hw'] = tuple(selected['image_hw'])
    calibration = CostCalibration(**selected)
    metadata = {key: value for key, value in payload.items() if key not in selected}
    metadata.update(hardware_id=calibration.hardware_id, image_hw=list(calibration.image_hw),
                    source_weight_binding='audit_only_reusable_across_seeds_same_architecture_and_mode')
    import hashlib
    metadata['cost_calibration_sha256'] = hashlib.sha256(raw).hexdigest()
    return calibration, metadata


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.dry_run = bool(args.dry_run or not args.run_training)
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
    if args.seed is not None:
        raw["training"]["seed"] = args.seed
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
              "seed": config.training.seed, "streaming_episode": True,
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
    from dust3r.training.reproducibility import seed_everything, append_jsonl
    from dust3r.training.provenance import (environment_contract, canonical_sha256,
        verify_actor_environment, source_provenance, runtime_backend_provenance,
        capture_rng_state, restore_rng_state, seed_loaded_opencv)
    from dust3r.online.streaming import DeviceSequence
    seed_everything(config.training.seed)
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
        from dust3r.training.validation import validate_hybrid_certification
        validate_hybrid_certification(checkpoint, file_sha256(config.training.data_manifest))
    configure_trainable_parameters(model, "r0")
    model.eval()
    hybrid_hash = file_sha256(args.hybrid_checkpoint) if args.hybrid_checkpoint else None
    contract = environment_contract(config, hashes, hybrid_hash)
    contract_hash = canonical_sha256(contract)
    actor = Actor(hidden=tuple(config.rl.actor_hidden)).to(args.device)
    critic = Critic(34 if config.rl.privileged_critic else 30, tuple(config.rl.critic_hidden)).to(args.device)
    normalizer = ObservationNormalizer().to(args.device)
    policy = PPOWritePolicy(actor, normalizer=normalizer)
    runner = model.make_online_runner(training=False, policy=policy)
    calibration = None
    if config.rl.cost_calibration_path:
        calibration, calibration_metadata = load_cost_calibration(
            config.rl.cost_calibration_path, config, args.device)
    else:
        calibration_metadata = None
    reward = DenseGTReward(quality_weight=config.rl.reward_quality_weight, write_weight=config.rl.reward_write_weight,
                           raft_weight=config.rl.reward_raft_weight, kda_geometry_weight=config.rl.reward_kda_geometry_weight,
                           error_cap=config.rl.reward_quality_error_cap, calibration=calibration)
    dataset = SequenceDataset(config.training.data_manifest, split="train",
                              image_hw=(config.training.image_height, config.training.image_width),
                              stride=config.training.stride)
    eligible = [index for index, (_, entries, _) in enumerate(dataset.samples) if len(entries) >= 4]
    if not eligible:
        raise ValueError("PPO actor training needs videos of at least four frames, including a nonforced decision")
    current_targets = None
    environment_id = contract_hash
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
    update_start = 0
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=True)
        if payload.get('schema') != 'endo3r_ppo_write_v1' or payload.get('environment_checkpoint_id') != environment_id:
            raise ValueError("PPO resume must use the same frozen reconstruction environment contract")
        verify_actor_environment(payload, config, hashes, hybrid_hash)
        if payload.get('config') != primitive_config(config):
            raise ValueError("PPO resume configuration changed; explicit new experiment required")
        if (payload.get('reward_calibration') != (None if calibration is None else vars(calibration)) or
                payload.get('reward_calibration_metadata') != calibration_metadata):
            raise ValueError('PPO resume reward/cost calibration changed; start an explicit new experiment.')
        optimizer.load_state_dict(payload['ppo'])
        normalizer.load_state_dict(payload['normalizer'])
        restore_rng_state(payload.get('rng_state'))
        update_start = int(payload['updates'])
        if update_start >= args.updates:
            raise ValueError("Resume already meets --updates; increase the explicit update budget")
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

    for update in range(update_start, args.updates):
        normalization_version = normalizer.freeze()
        buffer = RolloutBuffer(config.rl.rollout_transitions)
        while len(buffer) < buffer.capacity:
            if packet is None:
                seed_loaded_opencv(config.training.seed + sequence_cursor)
                sample = dataset.stream(eligible[sequence_cursor % len(eligible)])
                sequence_cursor += 1
                current_targets = DeviceSequence(sample, args.device, field='target')
                packet = environment.reset(DeviceSequence(sample, args.device), sample.episode_id)
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
        record = {"update": update, "seed": config.training.seed, "metrics": metrics}
        print(json.dumps(record))
        append_jsonl(args.log_jsonl, record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = source_provenance()
    rng = capture_rng_state()
    training_payload = {"schema": "endo3r_ppo_write_v1", "config": primitive_config(config),
                        'source_commit': source['git_commit'], 'source_provenance': source,
                        'legacy_audited_source_commit': SOURCE_COMMIT,
                        'external_revisions': runtime_revisions(),
                        'runtime_backend_provenance': runtime_backend_provenance(model),
                        "ppo": optimizer.state_dict(), "normalizer": normalizer.state_dict(),
                        "environment_checkpoint_id": environment_id, "original_checkpoint_hashes": hashes,
                        'environment_contract': contract, 'environment_contract_sha256': contract_hash,
                        'hybrid_checkpoint_sha256': hybrid_hash,
                        "observation_schema": "actor30_v1", "reward_calibration": None if calibration is None else vars(calibration),
                        'reward_calibration_metadata': calibration_metadata,
                        'rng_state': rng, 'torch_rng': rng['torch_cpu'], "updates": args.updates,
                        'resume_semantics': 'fresh_episode_complete_rng_restore_no_video_or_rollout_cursor',
                        "validated_for_deployment": False}
    torch.save(training_payload, output)
    actor_output = Path(args.actor_output)
    actor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value for key, value in training_payload.items() if key != "ppo"} |
               {"actor": actor.state_dict()}, actor_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
