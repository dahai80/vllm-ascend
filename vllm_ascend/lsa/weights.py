import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_projections(path: str, layer_ids: list[int], device: torch.device):
    # Distilled map: "<layer>.W_Q_idx" / "<layer>.W_K_idx", each [input_dim, index_dim] fp32
    if not os.path.exists(path):
        raise FileNotFoundError(f"lsa projections map not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    projections = {}
    for layer_id in layer_ids:
        q_key = f"{layer_id}.W_Q_idx"
        k_key = f"{layer_id}.W_K_idx"
        if q_key not in raw or k_key not in raw:
            raise KeyError(f"lsa projections map missing layer {layer_id}: has {sorted(raw.keys())[:4]}...")
        wq = raw[q_key].to(device=device, dtype=torch.float32)
        wk = raw[k_key].to(device=device, dtype=torch.float32)
        projections[layer_id] = (wq, wk)
    logger.info("LSA loaded projections for %d layers from %s", len(projections), path)
    return projections
