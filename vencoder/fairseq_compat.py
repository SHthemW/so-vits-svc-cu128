from contextlib import contextmanager
from functools import wraps

import torch
from fairseq import checkpoint_utils


@contextmanager
def _legacy_torch_load_for_fairseq():
    original_load = torch.load

    @wraps(original_load)
    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        yield
    finally:
        torch.load = original_load


def load_model_ensemble_and_task(*args, **kwargs):
    """Load trusted fairseq checkpoints with PyTorch 2.6+ compatibility."""
    with _legacy_torch_load_for_fairseq():
        return checkpoint_utils.load_model_ensemble_and_task(*args, **kwargs)
