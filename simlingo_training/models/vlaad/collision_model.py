"""
VLAAD collision/anomaly head, ported from carla_garage/team_code/vlaad/collision_model.py.

Only ``SupervisedAnomalyDetector`` is ported here: it is the frozen head that TransFuser++
consumes at train/inference time. It is pure PyTorch (no X-CLIP / transformers dependency),
so it can be instantiated inside the SimLingo training environment without pulling in the
offline extraction stack. The X-CLIP video embeddings are precomputed offline and stored as
``embeddings/NNNN.pt`` next to each ``rgb/NNNN.jpg`` (see
carla_garage/team_code/vlaad/extract_embeddings.py).

Contract (must match the carla_garage checkpoint exactly):
    forward(video_emb: [B, 768]) -> (projected: [B, 768], logit: [B] or scalar)
"""

import torch
import torch.nn as nn


class SupervisedAnomalyDetector(nn.Module):
  """Frozen collision-risk head over a 768-d X-CLIP video embedding.

  Kept bit-for-bit identical to the carla_garage definition so the saved
  ``anomaly_head_state_dict`` loads cleanly.
  """

  def __init__(self, input_dim=768, hidden_dim=256, use_uncertainty_weighting=True):
    super().__init__()
    self.projector = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, input_dim)
    )
    self.classifier = nn.Linear(input_dim, 1)
    self.use_uncertainty_weighting = use_uncertainty_weighting

    if use_uncertainty_weighting:
      self.log_var_sim = nn.Parameter(torch.zeros(1))
      self.log_var_cls = nn.Parameter(torch.zeros(1))
    else:
      self.sim_weight = 0.5
      self.cls_weight = 0.5

  def forward(self, video_emb):
    projected = self.projector(video_emb)
    logit = self.classifier(projected)
    return projected, logit.squeeze()
