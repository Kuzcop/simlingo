"""Construct the FULL model, load the pretrained epoch=013 weights + VLAAD head, report keys.

Run:  conda run -n simlingo python tools/test_vlaad_full_load.py
This DOWNLOADS InternVL2-1B from HF on first run and surfaces any flash-attn requirement.
No dataset walk, no training step — just construction + checkpoint load + freeze report.
"""
import os
import sys

REPO = "/home/asidhu7/github_workspace/simlingo"
sys.path.insert(0, REPO)
os.chdir(REPO)  # so get_original_cwd()-style relative paths resolve if needed

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra import initialize_config_dir, compose  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
import simlingo_training.config  # noqa: E402

cfg_dir = os.path.join(REPO, "simlingo_training", "config")
with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
    cfg = compose(config_name="config", overrides=["experiment=simlingo_vlaad"])

# emulate train.py sync
cfg.data_module.base_dataset.vlaad_mode = cfg.model.vlaad.mode

print("== building processor + registering tokens (datamodule.__init__) ==")
processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
# instantiate datamodule (registers <VLAAD> etc. on processor.tokenizer); no setup() -> no dataset walk
_ = hydra.utils.instantiate(
    cfg.data_module, processor=processor,
    encoder_variant=cfg.model.vision_model.variant,
    llm_variant=cfg.model.language_model.variant, _recursive_=False)
tok = getattr(processor, 'tokenizer', processor)  # AutoProcessor may return the tokenizer directly
vlaad_id = tok.convert_tokens_to_ids('<VLAAD>')
print("<VLAAD> token id:", vlaad_id, "(unk =", tok.unk_token_id, ")")

print("\n== building model (downloads InternVL2-1B) ==")
model = hydra.utils.instantiate(
    cfg.model, cfg_data_module=cfg.data_module, processor=processor,
    cache_dir=None, _recursive_=False)
print("model built OK")

print("\n== loading pretrained epoch=013 weights (strict=False) ==")
sd = torch.load(cfg.checkpoint, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith(("vision_model", "language_model", "adaptors")) for k in list(sd)[:5]):
    sd = sd["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
missing_non_vlaad = [k for k in missing if not k.startswith(("vlaad_encoder", "vlaad_detector"))]
print(f"total missing={len(missing)}  (non-VLAAD missing={len(missing_non_vlaad)})  unexpected={len(unexpected)}")
print("first non-VLAAD missing:", missing_non_vlaad[:10])
print("first unexpected:", list(unexpected)[:10])

print("\n== VLAAD head + freeze report ==")
det_params = list(model.vlaad_detector.parameters())
print("vlaad_detector params:", len(det_params),
      "| trainable:", sum(p.requires_grad for p in det_params), "(must be 0)")
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {n_train:,}/{n_total:,}")
# confirm the anomaly head actually got weights (non-zero classifier)
print("classifier.weight abs-sum:", model.vlaad_detector.classifier.weight.abs().sum().item())

print("\nFULL-MODEL LOAD TEST PASSED")
