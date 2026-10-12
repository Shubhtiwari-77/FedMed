"""Export the trained PyTorch MLP for the dependency-free Vercel function."""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.cdss import CKPT


def main() -> None:
    checkpoint = CKPT
    if not os.path.exists(checkpoint):
        raise SystemExit("Run python src/run_simulation.py before exporting the model.")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = saved["state_dict"]
    exported = {
        "diseases": cfg.DISEASES,
        "state_dict": {
            name: tensor.detach().cpu().tolist()
            for name, tensor in state_dict.items()
        },
    }
    output_path = os.path.join(cfg.OUT_DIR, "vercel_model.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(exported, file, separators=(",", ":"))
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()