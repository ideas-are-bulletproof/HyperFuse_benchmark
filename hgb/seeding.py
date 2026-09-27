"""
hgb/seeding.py
==============

One place to make a run reproducible.  ``set_global_seed`` seeds python,
numpy and (if present) torch + cudnn in exactly the way every upstream repo
here does in its own ``fix_seed`` (torch.manual_seed / cuda / cudnn
deterministic), so a method embedded at seed s inside this harness matches the
seeding it would get from its own train.py at seed s.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            # upstream repos set benchmark=True in their fix_seed; we match that
            # so we reproduce their numerical behaviour rather than diverge.
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
