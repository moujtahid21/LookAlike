# metrics.py
import torch
import torch.nn.functional as F
import numpy as np


def calculate_metrics(user_emb, db_emb):
    """
    Calculates distance metrics between two vectors.
    """
    # Convert to tensors if numpy
    if isinstance(user_emb, np.ndarray):
        user_emb = torch.from_numpy(user_emb)
    if isinstance(db_emb, np.ndarray):
        db_emb = torch.from_numpy(db_emb)

    # Ensure shape (1, D)
    if len(user_emb.shape) == 1: user_emb = user_emb.unsqueeze(0)
    if len(db_emb.shape) == 1: db_emb = db_emb.unsqueeze(0)

    # 1. Euclidean Distance (L2)
    l2_dist = F.pairwise_distance(user_emb, db_emb).item()

    # 2. Cosine Similarity (1 = Identical, -1 = Opposite)
    cosine_sim = F.cosine_similarity(user_emb, db_emb).item()

    # 3. Confidence Score (Heuristic based on L2)
    # Threshold 1.2 is roughly "no match"
    confidence = max(0, (1.2 - l2_dist) / 1.2) * 100

    return {
        "Euclidean Dist": l2_dist,
        "Cosine Sim": cosine_sim,
        "Confidence": confidence
    }