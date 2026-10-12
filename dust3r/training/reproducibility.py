"""Explicit experiment seeds; no dependency installation or device allocation."""
import random


def seed_everything(seed, deterministic=False):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic))
    # OpenCV is optional; seed it only if already loaded by a camera backend.
    import sys
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        cv2.setRNGSeed(seed % (2 ** 31 - 1))
    return {"seed": seed, "deterministic_algorithms": bool(deterministic)}


def sample_order(length, seed, epoch, shuffle=True):
    indices = list(range(length))
    if shuffle:
        random.Random(int(seed) + int(epoch)).shuffle(indices)
    return indices


def append_jsonl(path, record):
    if not path:
        return
    import json
    from pathlib import Path
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
