"""In-env smoke test for the FULL-model (simlingo_training) VLAAD integration.

Run:  conda run -n simlingo python tools/test_vlaad_full_smoke.py

Covers what is verifiable WITHOUT downloading the InternVL2-1B weights:
  1. Edited modules import (adaptors, custom_types, collision_model).
  2. Hydra composes experiment=simlingo_vlaad_tfpp and the VLAAD flags resolve;
     train.py's sync line works; the debug experiment defaults to vlaad.mode=off.
  3. Detector contract + VectorInputAdaptor token shape.
  4. Standalone reproduction of the internvl2 splice: the <VLAAD> position embedding is
     replaced and the sequence LENGTH is unchanged (the key safety property).
"""
import os
import sys

REPO = "/home/asidhu7/github_workspace/simlingo"
sys.path.insert(0, REPO)

print("== [1] importing edited modules ==")
import torch  # noqa: E402
from simlingo_training.models.vlaad.collision_model import SupervisedAnomalyDetector  # noqa: E402
from simlingo_training.models.adaptors.adaptors import VectorInputAdaptor  # noqa: E402
from simlingo_training.utils.custom_types import DrivingInput, DatasetOutput  # noqa: E402
assert "vlaad_embedding" in DrivingInput._fields and "vlaad_embedding" in DatasetOutput._fields
print("imports OK; DrivingInput/DatasetOutput carry vlaad_embedding")

print("\n== [2] Hydra compose ==")
from hydra import initialize_config_dir, compose  # noqa: E402
import simlingo_training.config  # noqa: E402  (register_configs runs on import)
cfg_dir = os.path.join(REPO, "simlingo_training", "config")
with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
    cfg = compose(config_name="config", overrides=["experiment=simlingo_vlaad_tfpp"])
print("model.vlaad:", cfg.model.vlaad)
assert cfg.model.vlaad.mode == "projected"
assert cfg.model.vlaad.trainable_scope == "heads_llm"
assert cfg.data_module.base_dataset.route_glob == "*/Town*"
assert cfg.data_module.base_dataset.holdout_town == "Town13"
assert cfg.data_module.dreamer_dataset is None
assert cfg.data_module.base_dataset.use_commentary is False and cfg.data_module.base_dataset.use_qa is False
# emulate train.py sync
cfg.data_module.base_dataset.vlaad_mode = cfg.model.vlaad.mode
assert cfg.data_module.base_dataset.vlaad_mode == "projected"
print("compose OK; driving-only + Town13 holdout + vlaad synced")

with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
    cfg_off = compose(config_name="config", overrides=["experiment=debug"])
assert cfg_off.model.vlaad.mode == "off"
print("debug experiment defaults vlaad.mode ->", cfg_off.model.vlaad.mode)

print("\n== [3] detector + adaptor ==")
B, EMB, HID = 3, 768, 896  # 896 = Qwen2-0.5B hidden size
det = SupervisedAnomalyDetector(EMB, 256, use_uncertainty_weighting=False).eval()
with torch.no_grad():
    projected, logit = det(torch.randn(B, EMB))
assert projected.shape == (B, EMB) and logit.shape == (B,)
enc_proj = VectorInputAdaptor(EMB, token_size=HID, hidden_size=256)
enc_logit = VectorInputAdaptor(1, token_size=HID, hidden_size=256)
assert enc_proj(projected).shape == (B, 1, HID)
assert enc_logit(logit.reshape(B, 1)).shape == (B, 1, HID)
print("detector + both adaptors OK -> one [B,1,896] token each")

print("\n== [4] splice reproduction (length-preserving) ==")
# Reproduce exactly the internvl2 block against a fake sequence.
VLAAD_ID = 99999
seq = 12
input_ids = torch.randint(0, 100, (B, seq))
input_ids[:, 7] = VLAAD_ID  # place a <VLAAD> token at position 7 in each row
inputs_embeds = torch.randn(B, seq, HID)
before = inputs_embeds.clone()
len_before = inputs_embeds.shape[1]

emb = torch.randn(B, EMB)
with torch.no_grad():
    projected, logit = det(emb)
vlaad_in = projected  # projected mode
vlaad_tokens = enc_proj(vlaad_in).to(inputs_embeds.dtype)
for b in range(B):
    positions = (input_ids[b] == VLAAD_ID).nonzero(as_tuple=True)[0]
    if positions.numel() > 0:
        inputs_embeds[b, positions[0]] = vlaad_tokens[b, 0]

assert inputs_embeds.shape[1] == len_before, "sequence length must be preserved"
# only position 7 changed
for b in range(B):
    changed = (inputs_embeds[b] != before[b]).any(dim=-1)
    assert changed[7].item() is True, "VLAAD position must change"
    assert changed.sum().item() == 1, "only the VLAAD position may change"
    assert torch.allclose(inputs_embeds[b, 7], vlaad_tokens[b, 0])
print("splice OK: exactly the <VLAAD> position replaced, length unchanged")

print("\nALL FULL-MODEL SMOKE CHECKS PASSED")
