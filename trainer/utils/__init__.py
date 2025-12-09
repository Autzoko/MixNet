from typing import Dict, Any
from pathlib import Path

import torch
import torch.nn as nn

import random
import numpy as np


def save_checkpoint(
        state: Dict[str, Any],
        out_dir: str,
        filename: str = "best_model.pth"
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_path / filename
    torch.save(state, checkpoint_path)
    print(F"    Checkpint saved: {checkpoint_path}")

def load_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"    Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"    Best val Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")

    return checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False