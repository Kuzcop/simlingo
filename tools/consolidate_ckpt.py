#!/usr/bin/env python3
"""Consolidate a Lightning/DeepSpeed ZeRO checkpoint into a single ``pytorch_model.pt``.

The CARLA agent (``team_code/agent_simlingo.py``) loads weights with a bare
``torch.load(path)`` (line 173) and derives the Hydra config from
``Path(path).parent.parent.parent/.hydra/config.yaml``. Training, however, writes DeepSpeed
ZeRO stage-2 **checkpoint directories** (``outputs/<run>/checkpoints/epoch=NNN.ckpt/``), which
``torch.load`` cannot read directly. This script consolidates the ZeRO shards into one fp32
state-dict saved as ``pytorch_model.pt`` *inside* the ``.ckpt`` directory -- exactly the layout
the published HF baseline uses (``.../epoch=013.ckpt/pytorch_model.pt``) and the same conversion
``simlingo_training/eval.py`` and ``train.py`` do via ``get_fp32_state_dict_from_zero_checkpoint``.

Accepts either:
  * a ``.ckpt`` DeepSpeed directory                -> writes ``<dir>/pytorch_model.pt``
  * a run directory (``outputs/<run>``)            -> picks the newest ``checkpoints/*.ckpt`` dir
  * an already-consolidated ``.pt`` file           -> no-op

Idempotent: skips if the target ``pytorch_model.pt`` already exists (unless ``--overwrite``).
Logs go to stderr; the final resolved ``.pt`` path is printed as the *only* line on stdout so a
shell can capture it:  ``CKPT_PT=$(python tools/consolidate_ckpt.py "$CKPT")``

Run in the ``simlingo`` conda env (needs torch + deepspeed).
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def log(*a):
  print(*a, file=sys.stderr, flush=True)


def newest_ckpt_dir(run_dir: Path) -> Path:
  """Return the checkpoint dir to consolidate under ``run_dir/checkpoints``.

  Prefers ``last.ckpt`` if present, else the highest-epoch ``epoch=*.ckpt`` directory.
  """
  ck = run_dir / "checkpoints"
  if not ck.is_dir():
    raise FileNotFoundError(f"no checkpoints/ under run dir: {run_dir}")
  last = ck / "last.ckpt"
  if last.is_dir():
    return last
  epochs = sorted(p for p in ck.glob("epoch=*.ckpt") if p.is_dir())
  if not epochs:
    raise FileNotFoundError(f"no epoch=*.ckpt or last.ckpt directories under {ck}")
  return epochs[-1]


def resolve_ckpt_dir(path: Path) -> Path:
  """Map a user-supplied path to the DeepSpeed .ckpt directory to consolidate."""
  if path.is_dir() and path.name.endswith(".ckpt"):
    return path
  if path.is_dir():
    return newest_ckpt_dir(path)
  raise FileNotFoundError(f"not a .ckpt dir or run dir: {path}")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("checkpoint", type=str,
                  help="a .ckpt DeepSpeed dir, a run dir (outputs/<run>), or a consolidated .pt file")
  ap.add_argument("--overwrite", action="store_true", help="reconsolidate even if pytorch_model.pt exists")
  args = ap.parse_args()

  path = Path(args.checkpoint).expanduser()
  if not path.exists():
    log(f"ERROR: path does not exist: {path}")
    sys.exit(1)

  # Already a consolidated .pt file -> nothing to do; echo it back.
  if path.is_file():
    log(f"[consolidate] already a file, using as-is: {path}")
    print(str(path))
    return

  ckpt_dir = resolve_ckpt_dir(path)
  out_pt = ckpt_dir / "pytorch_model.pt"

  if out_pt.exists() and not args.overwrite:
    log(f"[consolidate] exists, skipping (use --overwrite): {out_pt}")
    print(str(out_pt))
    return

  # Sanity: the agent expects .hydra/config.yaml at ckpt_dir.parent.parent/.hydra (== run dir).
  hydra_cfg = ckpt_dir.parent.parent / ".hydra" / "config.yaml"
  if not hydra_cfg.exists():
    log(f"WARNING: expected Hydra config not found at {hydra_cfg} -- the CARLA agent will fail to "
        f"build the model. Ensure the checkpoint lives at <run>/checkpoints/<epoch>.ckpt/.")

  log(f"[consolidate] reading ZeRO shards from: {ckpt_dir}")
  state_dict = get_fp32_state_dict_from_zero_checkpoint(str(ckpt_dir))
  log(f"[consolidate] consolidated {len(state_dict)} tensors; writing {out_pt}")
  torch.save(state_dict, str(out_pt))
  log("[consolidate] done.")
  print(str(out_pt))


if __name__ == "__main__":
  main()
