"""Uninstrumented online pipeline timing, separate from component profiling.

Only sequence/pair boundaries synchronize CUDA. Prediction cameras, memory
writes, RAFT and the controller are included. Input loading/GT preparation and
transfers are separately recorded; no synthetic result is an actual benchmark.
"""
import hashlib
import json
import math
import platform
import statistics
import time


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_actor_component_profile(profile, *, config, hardware_id,
                                     original_checkpoint_hashes,
                                     hybrid_checkpoint_sha256,
                                     actor_checkpoint_sha256,
                                     source_tree_sha256=None):
    """Validate an existing synchronized policy.act proxy without execution.

    This pure helper reads no files and imports no kernel/model packages.
    Profiling may use separate validation RGB; frame/GT identities are not
    required because the result is explicitly a component latency proxy.
    """
    if config.get("policy", {}).get("long_write_policy") != "ppo":
        raise ValueError("A separate actor component profile is valid only for the PPO write policy.")
    if profile.get("schema") != "endo3r_component_profile_v1":
        raise ValueError("Actor latency requires an endo3r_component_profile_v1 artifact.")
    if profile.get("timing_scope") != "synchronized_exclusive_components_not_headline_throughput":
        raise ValueError("Actor latency must come from separately synchronized exclusive component profiling.")
    if profile.get("component_operations", {}).get("policy") != "LongWritePolicy.act":
        raise ValueError("Component profile must explicitly identify policy as LongWritePolicy.act.")
    if not isinstance(profile.get("config"), dict) or canonical_hash(profile["config"]) != canonical_hash(config):
        raise ValueError("Actor component profile has a different full config.")
    if profile.get("config_sha256") is not None and profile["config_sha256"] != canonical_hash(profile["config"]):
        raise ValueError("Actor component profile config_sha256 does not match its embedded config.")
    if not hardware_id or profile.get("hardware", {}).get("hardware_id") != hardware_id:
        raise ValueError("Actor component profile hardware_id differs from headline benchmark hardware.")
    if not original_checkpoint_hashes or profile.get("original_checkpoint_hashes") != original_checkpoint_hashes:
        raise ValueError("Actor component profile original checkpoint hashes differ.")
    if profile.get("hybrid_checkpoint_sha256") != hybrid_checkpoint_sha256:
        raise ValueError("Actor component profile hybrid checkpoint hash differs.")
    if not actor_checkpoint_sha256 or profile.get("actor_checkpoint_sha256") != actor_checkpoint_sha256:
        raise ValueError("Actor component profile actor checkpoint hash differs or is missing.")
    profile_tree = profile.get("source_provenance", {}).get("source_tree_sha256")
    if profile_tree is not None and (not source_tree_sha256 or profile_tree != source_tree_sha256):
        raise ValueError("Actor component profile source_tree_sha256 differs or cannot be verified.")
    timing = profile.get("exclusive_component_timings", {}).get("policy", {})
    calls, mean_ms = timing.get("calls"), timing.get("mean_ms")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ValueError("Actor component profile needs at least one measured policy.act call.")
    if isinstance(mean_ms, bool) or not isinstance(mean_ms, (float, int)) or not math.isfinite(mean_ms) or mean_ms < 0:
        raise ValueError("Actor component profile policy.act mean_ms must be finite and nonnegative.")
    return {"actor_mean_latency_ms": float(mean_ms), "actor_profile_calls": calls,
            "actor_latency_source": "separate_validation_component_profile_proxy",
            "actor_latency_operation": "LongWritePolicy.act",
            "actor_source_tree_verified": profile_tree is not None,
            "actor_latency_is_headline_per_frame_measurement": False}


def runtime_metadata(device):
    import torch
    result = {"device": str(device), "torch": str(torch.__version__),
              "cuda_build": torch.version.cuda, "platform": platform.platform(),
              "gpu_name": None, "gpu_capability": None}
    device = torch.device(device)
    if device.type == "cuda":
        result.update(gpu_name=torch.cuda.get_device_name(device),
                      gpu_capability=list(torch.cuda.get_device_capability(device)))
    result["hardware_id"] = canonical_hash(result)
    return result


def latency_summary(values):
    values = sorted(values)
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {"mean_ms": statistics.mean(values), "p50_ms": statistics.median(values),
            "p95_ms": values[round(.95 * (len(values) - 1))]}


def benchmark_sequence(model, sample, config, device, *, runner_factory=None,
                       camera_factory=None, synchronize=None, clock=time.perf_counter,
                       pnp_seed=0):
    import torch
    from dust3r.online.streaming import create_stream_runner
    from dust3r.training.hybrid_trainer import move_tensors
    if runner_factory is None:
        runner_factory = create_stream_runner
    if camera_factory is None:
        from dust3r.evaluation.camera import PredictedCameraEstimator
        camera_factory = lambda: PredictedCameraEstimator(pnp_seed=pnp_seed)
    if synchronize is None:
        synchronize = (lambda: torch.cuda.synchronize(device)) if torch.device(device).type == "cuda" else (lambda: None)
    synchronize()
    setup_start = clock()
    runner = runner_factory(model, config, sample.frame_count, sample.episode_id)
    camera = camera_factory()
    synchronize()
    setup_ms = (clock() - setup_start) * 1000
    latencies, input_ms, frame_records = [], [], []
    output_normalized_latencies = []
    eligible, terminal_operations = [], []
    output_count = 0
    failure = None
    first_frame_ms = 0.
    synchronize()
    whole_start = clock()
    with torch.no_grad():
        for index in range(sample.frame_count):
            load_start = clock()
            frame, _ = sample.get_frame(index)
            frame = move_tensors(frame, device)
            synchronize()
            input_ms.append((clock() - load_start) * 1000)
            start = clock()
            try:
                pair = runner.step(frame, terminal=index == sample.frame_count - 1)
                if pair is None:
                    synchronize()
                    first_frame_ms += (clock() - start) * 1000
                    continue
                # Camera estimator accepts prediction only; GT cannot enter it.
                estimates = [camera.predict(pair.res1)]
                if pair.terminal:
                    estimates.append(camera.predict(pair.res2))
                if any(not estimate.valid for estimate in estimates):
                    reasons = [str(getattr(estimate, "error", "unknown failure")) for estimate in estimates if not estimate.valid]
                    raise FloatingPointError("Predicted camera/PnP failed; sequence timing is incomplete: " + "; ".join(reasons))
                output_count += len(estimates)
            except (FloatingPointError, ArithmeticError) as error:
                synchronize()
                failure = str(error)
                break
            synchronize()
            elapsed = (clock() - start) * 1000
            latencies.append(elapsed)
            output_normalized_latencies.extend([elapsed / len(estimates)] * len(estimates))
            flags = dict(getattr(runner, "last_commit_info", {}))
            record = {"candidate_index": pair.candidate_index,
                      "arrival_index": pair.arrival_index, "latency_ms": elapsed,
                      "output_count": len(estimates),
                      "amortized_output_latency_ms": elapsed / len(estimates),
                      "terminal": bool(pair.terminal),
                      "requested_write": bool(flags.get("requested_write")),
                      "successful_write": bool(flags.get("successful_write")),
                      "raft_called": bool(flags.get("raft_called")),
                      "kda_updated": bool(flags.get("kda_updated"))}
            frame_records.append(record)
            (terminal_operations if pair.terminal else eligible).append(record)
            # No prediction/pointmap list is retained on the GPU.
            del pair, estimates
    if failure is None and (output_count != sample.frame_count or len(latencies) != sample.frame_count - 1):
        failure = (f"Incomplete output coverage: expected {sample.frame_count} outputs/{sample.frame_count - 1} pairs, "
                   f"got {output_count}/{len(latencies)}.")
    synchronize()
    whole_ms = (clock() - whole_start) * 1000
    pipeline_ms = sum(latencies)
    def rate(key):
        return sum(row[key] for row in eligible) / len(eligible) if eligible else None
    memory = getattr(model, "hybrid_memory", None)
    state = getattr(runner, "state", None)
    return {"episode_id": sample.episode_id, "input_frames": sample.frame_count,
            "output_frames": output_count, "pair_count": len(latencies),
            "status": "failed" if failure else "completed", "failure": failure,
            "pipeline_ms": pipeline_ms, "stream_load_transfer_and_pipeline_ms": whole_ms,
            "sequence_memory_camera_setup_ms": setup_ms, "initial_frame_setup_ms": first_frame_ms,
            "pipeline_with_setup_ms": pipeline_ms + setup_ms + first_frame_ms,
            "fps_with_sequence_setup": output_count * 1000 / (pipeline_ms + setup_ms + first_frame_ms)
                                       if pipeline_ms + setup_ms + first_frame_ms > 0 and failure is None else None,
            "input_preparation_ms": sum(input_ms),
            "fps": output_count * 1000 / pipeline_ms if pipeline_ms > 0 and failure is None else None,
            **latency_summary(latencies),
            "latency_unit": "pair_operation_including_terminal_two_outputs",
            "pair_ms": latency_summary(latencies),
            "amortized_output_ms": latency_summary(output_normalized_latencies),
            "amortized_output_latency_definition": "each_pair_latency_divided_by_its_output_count_repeated_per_output_not_observed_frame_arrival_latency",
            "frame_timings": frame_records,
            "eligible_decisions": len(eligible), "write_rate": rate("successful_write"),
            "requested_write_rate": rate("requested_write"), "raft_call_rate": rate("raft_called"),
            "kda_update_rate": rate("kda_updated"),
            "total_raft_calls": sum(row["raft_called"] for row in frame_records),
            "terminal_operations": terminal_operations,
            "kda_state_bytes": state.kda_state.numel() * state.kda_state.element_size() if hasattr(state, "kda_state") else None,
            "actual_backends": {"short": memory.short.last_backend if memory is not None else "legacy_dense",
                                "long": memory.long.adapter.last_backend if memory is not None else "legacy_dense"},
            "actor_mean_ms": None}


def aggregate_performance(sequences):
    if not sequences:
        raise ValueError("Performance aggregation needs at least one sequence.")
    completed = [sequence for sequence in sequences if sequence["status"] == "completed"]
    latencies = [frame["latency_ms"] for sequence in completed for frame in sequence["frame_timings"]]
    total_time = sum(sequence["pipeline_ms"] for sequence in completed)
    total_frames = sum(sequence["output_frames"] for sequence in completed)
    eligible = [row for sequence in completed for row in sequence["frame_timings"] if not row["terminal"]]
    normalized_latencies = [frame["amortized_output_latency_ms"]
                            for sequence in completed for frame in sequence["frame_timings"]
                            for _ in range(frame["output_count"])]
    complete = len(completed) == len(sequences)
    measured_fps = total_frames * 1000 / total_time if total_time > 0 else None
    return {"completed_sequences": len(completed), "failed_sequences": len(sequences) - len(completed),
            "coverage_complete": complete,
            "fps": measured_fps if complete else None,
            "successful_sequences_only_fps": measured_fps,
            "timed_output_frames": total_frames, "timed_pipeline_ms": total_time,
            **latency_summary(latencies),
            "latency_unit": "pair_operation_including_terminal_two_outputs",
            "pair_ms": latency_summary(latencies), "amortized_output_ms": latency_summary(normalized_latencies),
            "write_rate": sum(row["successful_write"] for row in eligible) / len(eligible) if eligible else None,
            "raft_call_rate": sum(row["raft_called"] for row in eligible) / len(eligible) if eligible else None,
            "kda_update_rate": sum(row["kda_updated"] for row in eligible) / len(eligible) if eligible else None,
            "actor_mean_ms": None}
