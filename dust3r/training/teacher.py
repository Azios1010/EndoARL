"""Optional prefix-aligned, detached teacher traces; never downloads models."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import torch


TRACE_SCHEMA = "endo3r_teacher_prefix_v1"


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required local checkpoint/file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TeacherTraceMetadata:
    episode_id: str
    checkpoint_hash: str
    preprocessing_hash: str
    action_trajectory_hash: str
    source_commit: str = "8145c8e5535d2111974561e7d1cc06f1d008fe5e"
    schema: str = TRACE_SCHEMA


class TeacherTraceCache:
    """Teacher follows its own official causal query/write trajectory.

    Traces align prefixes and patch positions, not student numerical queries.
    Metadata mismatch or nonfinite teachers fail instead of changing baseline.
    """

    def __init__(self, directory, expected: TeacherTraceMetadata):
        self.directory = Path(directory)
        self.expected = expected

    def _path(self, candidate_index):
        episode_key = hashlib.sha256(self.expected.episode_id.encode("utf-8")).hexdigest()[:24]
        return self.directory / episode_key / f"prefix_{int(candidate_index):06d}.pt"

    def save(self, candidate_index, arrival_index, feature):
        if arrival_index != candidate_index + 1:
            raise ValueError("Teacher trace must follow Endo3R arrival-time causality")
        if feature.ndim != 3 or not torch.isfinite(feature).all():
            raise ValueError("Teacher feature must be finite [B,P,C]")
        path = self._path(candidate_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": vars(self.expected), "candidate_index": candidate_index,
                    "arrival_index": arrival_index, "feature": feature.detach().float().cpu()}, path)

    def load(self, candidate_index, arrival_index, device=None):
        path = self._path(candidate_index)
        if not path.is_file():
            raise FileNotFoundError(f"Teacher trace is required locally, no download fallback: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("metadata") != vars(self.expected):
            raise ValueError(f"Teacher trace provenance/config mismatch: {path}")
        if payload.get("arrival_index") != arrival_index or payload.get("candidate_index") != candidate_index:
            raise ValueError("Teacher trace prefix does not match student causal prefix")
        feature = payload["feature"]
        if feature.ndim != 3 or not torch.isfinite(feature).all():
            raise ValueError("Teacher trace is nonfinite or has an invalid shape")
        return feature.to(device=device)


def warm_start_weight(step, total_steps, initial_weight=0.1, anneal_fraction=0.1):
    if total_steps <= 0 or not 0 < anneal_fraction <= 1 or initial_weight < 0:
        raise ValueError("Teacher schedule requires positive steps/fraction and nonnegative weight")
    return initial_weight * max(0.0, 1.0 - step / max(1.0, total_steps * anneal_fraction))
