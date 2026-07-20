#!/bin/bash
# ---------------------------------------------------------------------------
# VLAAD x SimLingo fine-tuning ablation launcher (local, non-SLURM).
#
# Fine-tunes the FULL SimLingo model (InternVL2-1B) from the published
# epoch=013 weights on the TF++ dataset (Town13 holdout) with the VLAAD
# collision signal, sweeping the trainable scope and injection mode.
#
# Pick a case with the TEST_CASE variable below, or pass it as arg 1:
#     ./train_vlaad_ablation.sh 2
#
# Cases (see the table in the plan / chat):
#   0  smoke     1-GPU fast_dev_run, full + logit    (wiring check, ~2 steps)
#   -- priority 1: the decisive comparison --
#   1  full        + off      (baseline: TF++ fine-tune, no VLAAD)
#   2  full        + logit
#   -- priority 2 --
#   3  heads_llm   + off
#   4  heads_llm   + logit
#   -- priority 3 --
#   5  llm         + off
#   6  llm         + logit
#   -- priority 4 --
#   7  heads       + off
#   8  heads       + logit
#   -- injection comparison at the winning scope (edit WINNING_SCOPE) --
#   9  <WINNING_SCOPE> + projected (numeric)
#   10 <WINNING_SCOPE> + text (verbalized)
# ---------------------------------------------------------------------------
set -euo pipefail

# ===== EDIT ME =====
TEST_CASE="${1:-2}"          # default case if no arg given
WINNING_SCOPE="full"         # scope used by cases 9 & 10 (set after cases 1-8)
CONDA_ENV="simlingo"
WORK_DIR="/home/asidhu7/github_workspace/simlingo"
WANDB_PROJECT_NAME="VLAAD"   # same project as TF++ (team_code/shell_train.sh) so both live together
WANDB_GROUP="Simlingo"   # groups every run below in the wandb UI
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Set the GPUs to use
# ===================

export NUM_GPUS=$(awk -F, '{print ($1=="") ? 0 : NF}' <<< "$CUDA_VISIBLE_DEVICES")
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' selects no GPUs; set it to e.g. 1,2,3." >&2; exit 1
fi
export OMP_NUM_THREADS=$(( 64 / NUM_GPUS ))  # Limits pytorch to spawn at most num cpus cores threads (64)


# ---- environment ----
source ~/.bashrc
cd "${WORK_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${WORK_DIR}"
export MASTER_ADDR=localhost
export OPENBLAS_NUM_THREADS=1        # no numpy multithreading
export WANDB__SERVICE_WAIT=300
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1 
export NCCL_DEBUG=INFO

# ---- per-case configuration ----
SCOPE=""; MODE=""; INJECTION=""; NAME=""; EXTRA=""
case "${TEST_CASE}" in
  0)  SCOPE=full;      MODE=logit;     INJECTION=numeric; NAME=smoke
      EXTRA="gpus=1 debug=True fast_dev_run=2 strategy=auto data_module.num_workers=0 data_module.batch_size=2"
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} ;;  # first GPU from your selection
  1)  SCOPE=full;      MODE=off;                          NAME=full_off ;;
  2)  SCOPE=full;      MODE=logit;     INJECTION=numeric; NAME=full_logit ;;
  3)  SCOPE=heads_llm; MODE=off;                          NAME=heads_llm_off ;;
  4)  SCOPE=heads_llm; MODE=logit;     INJECTION=numeric; NAME=heads_llm_logit ;;
  5)  SCOPE=llm;       MODE=off;                          NAME=llm_off ;;
  6)  SCOPE=llm;       MODE=logit;     INJECTION=numeric; NAME=llm_logit ;;
  7)  SCOPE=heads;     MODE=off;                          NAME=heads_off ;;
  8)  SCOPE=heads;     MODE=logit;     INJECTION=numeric; NAME=heads_logit ;;
  9)  SCOPE="${WINNING_SCOPE}"; MODE=projected; INJECTION=numeric; NAME="${WINNING_SCOPE}_projected" ;;
  10) SCOPE="${WINNING_SCOPE}"; MODE=logit;     INJECTION=text;    NAME="${WINNING_SCOPE}_text" ;;
  *)  echo "Unknown TEST_CASE='${TEST_CASE}'. Valid: 0-10 (see header)." >&2; exit 1 ;;
esac

# gpus: case 0 sets its own (gpus=1) in EXTRA; everything else uses $NUM_GPUS
GPU_ARG=""
if [[ "${TEST_CASE}" != "0" ]]; then GPU_ARG="gpus=${NUM_GPUS}"; fi

# injection is only meaningful when the signal is on (mode != off)
INJ_ARG=""
if [[ "${MODE}" != "off" && -n "${INJECTION}" ]]; then INJ_ARG="model.vlaad.injection=${INJECTION}"; fi

# wandb: group all runs together; primary tag == NAME == output dir (outputs/${name}),
# plus a secondary injection tag for filtering. Hydra list syntax (no spaces): wandb_tags=[a,b,c]
TAGS="${NAME}"
if [[ "${MODE}" != "off" && -n "${INJECTION}" ]]; then TAGS="${TAGS},inj-${INJECTION}"; fi
WANDB_ARGS="wandb_project=${WANDB_PROJECT_NAME} wandb_group=${WANDB_GROUP} wandb_tags=[${TAGS}]"

echo "=========================================================="
echo " TEST_CASE=${TEST_CASE}  ->  scope=${SCOPE}  mode=${MODE}  injection=${INJECTION:-n/a}"
echo " run name : ${NAME}"
echo " wandb    : project=${WANDB_PROJECT_NAME}  group=${WANDB_GROUP}  tags=[${TAGS}]"
echo "=========================================================="

# shellcheck disable=SC2086
python simlingo_training/train.py experiment=simlingo_vlaad \
  model.vlaad.trainable_scope="${SCOPE}" \
  model.vlaad.mode="${MODE}" \
  ${INJ_ARG} \
  name="${NAME}" \
  ${GPU_ARG} \
  ${WANDB_ARGS} \
  ${EXTRA}
