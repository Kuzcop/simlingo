#!/bin/bash
#
# Parallel (multi-GPU), multi-run local evaluation of a SimLingo agent on the Town13 VALIDATION
# routes -- the SimLingo analogue of carla_garage's
# leaderboard/run_leaderboard_ensemble_parallel_local.sh.
#
# Splits leaderboard/data/routes_validation.xml (20 routes, all Town13) across N GPUs by route index
# (one CARLA server + one leaderboard evaluator per GPU), and repeats the whole eval NUM_RUNS times
# into run_1..run_N so the runs can be averaged for a mean +/- SE report (CARLA is non-deterministic;
# the traffic-manager seed is held fixed, matching the TF++ Table-2 methodology). Each run's shards
# are auto-aggregated (LB2.0 + LB2.1 scoring) into <run>/results_merged.json.
#
# Output layout (VAL_ROOT/<experiment>/, <experiment> == the checkpoint's run/session name):
#   <experiment>/run_1/{logs,checkpoints,results_merged.json}
#   <experiment>/run_2/{logs,checkpoints,results_merged.json}
#   <experiment>/run_3/{logs,checkpoints,results_merged.json}
#
# CHECKPOINT may be any of:
#   * a consolidated .pt file      (e.g. the HF baseline .../epoch=013.ckpt/pytorch_model.pt) -> used directly
#   * a DeepSpeed .ckpt directory  (from training)  -> consolidated to <dir>/pytorch_model.pt first
#   * a run directory (outputs/<run>)               -> newest checkpoints/*.ckpt consolidated first
# In every case the agent (team_code/agent_simlingo.py) derives its Hydra model config from
# <ckpt>.parent.parent.parent/.hydra/config.yaml, so the checkpoint MUST sit at
# <run>/checkpoints/<epoch>.ckpt/pytorch_model.pt with a sibling <run>/.hydra/config.yaml.
#
# Override any variable inline, e.g.:
#   CHECKPOINT=/home/asidhu7/github_workspace/simlingo/outputs/full_off \
#   GPUS="0 1 2 3" NUM_RUNS=1 bash run_simlingo_town13_eval_parallel.sh
#
# Re-launching the same command resumes: finished runs are skipped (SKIP_COMPLETE=1); a partially-done
# run resumes its unfinished shards via --resume=1. DRY_RUN=1 prints the plan and exits.
# Assumes NO CARLA server is running yet; this script launches one per GPU per run and tears them down.
#
# NOTE: intentionally NOT using `set -e` -- we background many evaluators and want to `wait` on each
# and report its exit code rather than abort on the first failure. Preflight checks exit explicitly.

# ===== EDIT ME =====
: "${CONDA_ENV:=simlingo}"
# WORK_DIR is derived from this script's location (NOT `:=`, so a stale WORK_DIR inherited from the
# environment -- e.g. pointing at carla_garage -- cannot hijack it). This script lives in the simlingo repo.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CARLA_ROOT:=/storage/aman/carla}"       # client PythonAPI root (server binary lives here too)
: "${CARLA_SIM_ROOT:=/storage/aman/carla}"   # where CarlaUE4.sh lives (the simulator install)
# Default target: the published HF baseline SimLingo (epoch=013). Point at an outputs/<run> to eval a fine-tune.
: "${CHECKPOINT:=/home/asidhu7/.cache/huggingface/hub/models--RenzKa--simlingo/snapshots/26c7c89e797d4e25bbf640013317af8da26a5454/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
: "${GPUS:=0 1 2 3 4 5 6 7}"                 # space-separated physical GPU ids
: "${NUM_RUNS:=3}"                           # repeat into run_1..run_N (mean +/- SE)
: "${VAL_ROOT:=/storage/aman/simlingo/town13_val}"
# ===================

# ---- environment ----
source ~/.bashrc 2>/dev/null || true
# Activate the simlingo conda env (carla 0.9.15 is pip-installed there; no PythonAPI egg needed).
# `conda activate` is unreliable in non-interactive/headless shells (the conda shell function may be
# half-initialized by ~/.bashrc), so we ALSO resolve the env's python explicitly as PYBIN and launch
# every subprocess with "$PYBIN" -- this does not depend on PATH/activation propagating.
CONDA_BASE=""
for _c in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_c/etc/profile.d/conda.sh" ] && { CONDA_BASE="$_c"; source "$_c/etc/profile.d/conda.sh"; break; }
done
if [ -z "$CONDA_BASE" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV}" 2>/dev/null || true

# Resolve the env python explicitly (bulletproof against activation not propagating to subprocesses).
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ] && [[ "${CONDA_PREFIX}" == *"${CONDA_ENV}"* ]]; then
    PYBIN="${CONDA_PREFIX}/bin/python"
elif [ -n "${CONDA_BASE}" ] && [ -x "${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" ]; then
    PYBIN="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
else
    PYBIN="python3"
    echo "WARN: could not resolve the '${CONDA_ENV}' env python; falling back to 'python3' (may lack deps)."
fi
echo "Using python: ${PYBIN}"
cd "${WORK_DIR}"

export CARLA_ROOT
export WORK_DIR
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
# carla is imported from the conda env (site-packages); PYTHONPATH only needs the repo + LB/SR + team_code.
export PYTHONPATH="${WORK_DIR}":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH:-}

# ---- agent / evaluator config ----
: "${TEAM_AGENT:=${WORK_DIR}/team_code/agent_simlingo.py}"
: "${EVALUATOR:=${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py}"
: "${AGGREGATOR:=${LEADERBOARD_ROOT}/scripts/aggregate_ensemble_eval.py}"
: "${ROUTES:=${LEADERBOARD_ROOT}/data/routes_validation.xml}"   # 20 routes, all Town13

# ---- run config ----
: "${SKIP_COMPLETE:=1}"      # 1: skip a run whose results_merged.json already has a global_record
: "${RUN_STAGGER:=10}"       # seconds between runs (let CARLA ports/GPUs settle)
: "${RESUME:=1}"
: "${DEBUG_CHALLENGE:=0}"
: "${TM_SEED:=100}"          # traffic-manager seed (leaderboard default; same for all runs)
: "${TIMEOUT:=300}"          # CARLA client timeout (s)
: "${AUTO_AGGREGATE:=1}"     # 1: run the shard aggregator after each run's GPUs finish
: "${DRY_RUN:=0}"            # 1: print plan (subsets/ports/commands) and exit before launching

read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# MANUAL_SUBSETS: optional space-separated "start-end" per GPU (positionally matched to GPUS). Use the
# literal '' to skip a GPU. Empty -> auto-distribute evenly (last GPU absorbs the remainder).
: "${MANUAL_SUBSETS:=}"
read -r -a MANUAL_LIST <<< "$MANUAL_SUBSETS"

# --- Resolve the checkpoint to a single consolidated .pt (consolidating a DeepSpeed dir if needed) ---
if [ ! -e "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT does not exist: $CHECKPOINT" >&2
    exit 1
fi
if [ -f "$CHECKPOINT" ]; then
    AGENT_CONFIG="$CHECKPOINT"
else
    echo "CHECKPOINT is a directory -> consolidating DeepSpeed shards to pytorch_model.pt ..."
    AGENT_CONFIG=$("$PYBIN" "${WORK_DIR}/tools/consolidate_ckpt.py" "$CHECKPOINT" | tail -n 1)
    if [ -z "$AGENT_CONFIG" ] || [ ! -f "$AGENT_CONFIG" ]; then
        echo "ERROR: consolidation failed; no pytorch_model.pt produced for $CHECKPOINT" >&2
        exit 1
    fi
fi

# The agent reads the Hydra config from <ckpt>.parent.parent.parent/.hydra/config.yaml.
RUN_ROOT=$(cd "$(dirname "$AGENT_CONFIG")/../.." && pwd)   # == <run> dir (session)
HYDRA_CFG="${RUN_ROOT}/.hydra/config.yaml"
if [ ! -f "$HYDRA_CFG" ]; then
    echo "ERROR: agent needs the Hydra config at $HYDRA_CFG but it is missing." >&2
    echo "       The checkpoint must live at <run>/checkpoints/<epoch>.ckpt/pytorch_model.pt." >&2
    exit 1
fi
: "${EXPERIMENT_NAME:=$(basename "$RUN_ROOT")}"
: "${RESULTS_DIR:=${VAL_ROOT}/${EXPERIMENT_NAME}}"

# --- Preflight ---
for f in "$TEAM_AGENT" "$EVALUATOR" "$AGGREGATOR" "$ROUTES"; do
    [ -f "$f" ] || { echo "ERROR: required file not found: $f" >&2; exit 1; }
done
[ -x "$CARLA_SIM_ROOT/CarlaUE4.sh" ] || { echo "ERROR: CarlaUE4.sh not found/executable at $CARLA_SIM_ROOT" >&2; exit 1; }

mkdir -p "$RESULTS_DIR"

# --- Compute per-GPU route subsets ("start-end" id ranges) ---
TOTAL_ROUTES=$(grep -c '<route id=' "$ROUTES")

declare -a SUBSETS
if [ "${#MANUAL_LIST[@]}" -gt 0 ]; then
    echo "Using manual route subsets"
    for ((i = 0; i < NUM_GPUS; i++)); do
        entry="${MANUAL_LIST[$i]}"        # a literal '' entry means "skip this GPU"
        [ "$entry" = "''" ] && entry=""
        SUBSETS[$i]="$entry"
    done
else
    echo "Auto-distributing ${TOTAL_ROUTES} routes across ${NUM_GPUS} GPUs"
    ROUTES_PER_GPU=$((TOTAL_ROUTES / NUM_GPUS))
    for ((i = 0; i < NUM_GPUS; i++)); do
        START=$((i * ROUTES_PER_GPU))
        END=$(((i + 1) * ROUTES_PER_GPU - 1))
        if [ "$i" -eq $((NUM_GPUS - 1)) ]; then
            END=$((TOTAL_ROUTES - 1))
        fi
        SUBSETS[$i]="${START}-${END}"
    done
fi

echo "=========================================="
echo "SimLingo Town13 Evaluation (parallel, ${NUM_GPUS} GPUs x ${NUM_RUNS} runs)"
echo "=========================================="
echo "Agent:        $TEAM_AGENT"
echo "Checkpoint:   $AGENT_CONFIG"
echo "Hydra config: $HYDRA_CFG"
echo "Routes:       $ROUTES ($TOTAL_ROUTES total, Town13)"
echo "Experiment:   $EXPERIMENT_NAME"
echo "Results dir:  $RESULTS_DIR  (runs -> run_1..run_${NUM_RUNS})"
echo "TM seed:      $TM_SEED"
echo "Route distribution:"
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -z "${SUBSETS[$i]}" ]; then
        echo "  GPU ${GPU_LIST[$i]}: SKIPPED"
    else
        echo "  GPU ${GPU_LIST[$i]}: routes ${SUBSETS[$i]}"
    fi
done
echo "=========================================="
echo ""

# --- Helpers ---
find_free_port_range() {   # a free port from a fixed range (mirrors the TF++ / data-collection scripts)
    local start=$1 end=$2
    comm -23 <(seq "$start" "$end" | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1
}
subset_ckpt() { echo "${CKPT_DIR}/results_gpu${1}_${2//-/_}.json"; }   # uses per-run CKPT_DIR

run_is_complete() {   # $1 = run dir; true iff its results_merged.json has a populated global_record
    local m="$1/results_merged.json"
    [ -s "$m" ] || return 1
    "$PYBIN" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('_checkpoint',{}).get('global_record') else 1)" "$m" 2>/dev/null
}

if [ "$DRY_RUN" = "1" ]; then
    CKPT_DIR="${RESULTS_DIR}/run_1/checkpoints"
    echo "DRY_RUN=1 — planned per-GPU evaluator commands for run_1 (repeats into run_1..run_${NUM_RUNS}):"
    for ((i = 0; i < NUM_GPUS; i++)); do
        SUBSET="${SUBSETS[$i]}"
        [ -z "$SUBSET" ] && continue
        GPU="${GPU_LIST[$i]}"
        CARLA_P=$(find_free_port_range 20000 20400); TM_P=$(find_free_port_range 30000 30400); ST_P=$(find_free_port_range 10000 10400)
        CKPT=$(subset_ckpt "$GPU" "$SUBSET")
        echo ""
        echo "  # GPU $GPU: routes $SUBSET  (CARLA=$CARLA_P, TM=$TM_P, stream=$ST_P)"
        echo "  SAVE_PATH=${RESULTS_DIR}/run_1/agent_saves/ CUDA_VISIBLE_DEVICES=$GPU ${PYBIN} $EVALUATOR \\"
        echo "      --routes=$ROUTES --routes-subset=$SUBSET \\"
        echo "      --checkpoint=$CKPT \\"
        echo "      --agent=$TEAM_AGENT --agent-config=$AGENT_CONFIG \\"
        echo "      --debug=$DEBUG_CHALLENGE --resume=$RESUME --timeout=$TIMEOUT \\"
        echo "      --port=$CARLA_P --traffic-manager-port=$TM_P --traffic-manager-seed=$TM_SEED"
    done
    echo ""
    echo "(DRY_RUN) Each run is aggregated into <run>/results_merged.json via:"
    echo "  ${PYBIN} $AGGREGATOR --checkpoints-dir <run>/checkpoints -e <run>/results_merged.json"
    exit 0
fi

# --- Teardown ---
declare -a CARLA_PIDS AGENT_PIDS
kill_carla_servers() {   # stop the current run's CARLA servers and reset the PID arrays
    for pid in "${CARLA_PIDS[@]}"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    CARLA_PIDS=(); AGENT_PIDS=()
}
cleanup() {   # on any exit (incl. Ctrl-C), stop whatever is currently running
    echo ""
    echo "Cleaning up: stopping evaluators and CARLA servers..."
    for pid in "${AGENT_PIDS[@]}"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
    done
    kill_carla_servers
}
trap cleanup EXIT

# --- One full parallel eval into $RUN_DIR (globals: RUN_DIR, LOG_DIR, CKPT_DIR, SAVE_DIR) ---
# Returns 0 on success, 1 if a CARLA server failed to come up.
run_one() {
    CARLA_PIDS=(); AGENT_PIDS=()
    declare -a CARLA_PORTS TM_PORTS STREAM_PORTS AGENT_GPU AGENT_SUBSET

    # Pick fresh, collision-free ports for each active GPU.
    for ((i = 0; i < NUM_GPUS; i++)); do
        [ -z "${SUBSETS[$i]}" ] && continue
        STREAM_PORTS[$i]=$(find_free_port_range 10000 10400)
        CARLA_PORTS[$i]=$(find_free_port_range 20000 20400)
        TM_PORTS[$i]=$(find_free_port_range 30000 30400)
    done

    # Phase 1: launch one CARLA server per active GPU.
    echo "=== [run ${RUN_IDX}] Phase 1: launching CARLA servers ==="
    for ((i = 0; i < NUM_GPUS; i++)); do
        [ -z "${SUBSETS[$i]}" ] && continue
        GPU="${GPU_LIST[$i]}"
        echo "GPU $GPU: CARLA=${CARLA_PORTS[$i]}, TM=${TM_PORTS[$i]}, stream=${STREAM_PORTS[$i]}"
        "$CARLA_SIM_ROOT/CarlaUE4.sh" \
            -world-port="${CARLA_PORTS[$i]}" \
            -RenderOffScreen \
            -nosound \
            -graphicsadapter="$GPU" \
            -carla-streaming-port="${STREAM_PORTS[$i]}" > "${LOG_DIR}/carla_gpu${GPU}.log" 2>&1 &
        CARLA_PIDS[$i]=$!
        echo "  CARLA PID: ${CARLA_PIDS[$i]}"
        sleep 5
    done

    # Poll each server (up to ~120s) until its RPC port accepts connections; fail fast if it died.
    echo ""
    echo "Waiting for CARLA servers to come up..."
    for ((i = 0; i < NUM_GPUS; i++)); do
        [ -z "${SUBSETS[$i]}" ] && continue
        GPU="${GPU_LIST[$i]}"; PORT="${CARLA_PORTS[$i]}"
        echo -n "  GPU $GPU (port $PORT): "
        up=0
        for _ in $(seq 1 60); do
            if ss -tuln | grep -q ":$PORT "; then up=1; break; fi
            if ! kill -0 "${CARLA_PIDS[$i]}" 2>/dev/null; then
                echo "DIED during startup — see ${LOG_DIR}/carla_gpu${GPU}.log" >&2
                return 1
            fi
            sleep 2
        done
        if [ "$up" = "1" ]; then echo "up"; else
            echo "TIMEOUT — see ${LOG_DIR}/carla_gpu${GPU}.log" >&2
            return 1
        fi
    done
    sleep 5  # small grace period after the ports open

    # Phase 2: launch one evaluator per active GPU.
    echo ""
    echo "=== [run ${RUN_IDX}] Phase 2: launching evaluators ==="
    for ((i = 0; i < NUM_GPUS; i++)); do
        SUBSET="${SUBSETS[$i]}"
        [ -z "$SUBSET" ] && continue
        GPU="${GPU_LIST[$i]}"
        CKPT=$(subset_ckpt "$GPU" "$SUBSET")
        echo "GPU $GPU: routes $SUBSET -> $(basename "$CKPT")"
        # SAVE_PATH is REQUIRED by agent_simlingo.py (it unconditionally concatenates os.environ['SAVE_PATH']).
        SAVE_PATH="${SAVE_DIR}/" \
        DEBUG_CHALLENGE="$DEBUG_CHALLENGE" \
        CUDA_VISIBLE_DEVICES=$GPU \
        "$PYBIN" "$EVALUATOR" \
            --routes="$ROUTES" \
            --routes-subset="$SUBSET" \
            --checkpoint="$CKPT" \
            --agent="$TEAM_AGENT" \
            --agent-config="$AGENT_CONFIG" \
            --debug="$DEBUG_CHALLENGE" \
            --resume="$RESUME" \
            --timeout="$TIMEOUT" \
            --port="${CARLA_PORTS[$i]}" \
            --traffic-manager-port="${TM_PORTS[$i]}" \
            --traffic-manager-seed="$TM_SEED" > "${LOG_DIR}/agent_gpu${GPU}.log" 2>&1 &
        AGENT_PIDS[$i]=$!
        AGENT_GPU[$i]="$GPU"; AGENT_SUBSET[$i]="$SUBSET"
        echo "  agent PID: ${AGENT_PIDS[$i]}"
        sleep 2
    done

    echo ""
    echo "All evaluators for run ${RUN_IDX} launched. Logs: ${LOG_DIR}/agent_gpu*.log"
    echo ""

    # Wait for all evaluators; report per-GPU exit codes.
    local failures=0
    for ((i = 0; i < NUM_GPUS; i++)); do
        [ -z "${AGENT_PIDS[$i]}" ] && continue
        wait "${AGENT_PIDS[$i]}"; code=$?
        AGENT_PIDS[$i]=""   # already reaped; don't re-kill in cleanup
        if [ "$code" -eq 0 ]; then
            echo "OK  GPU ${AGENT_GPU[$i]} (routes ${AGENT_SUBSET[$i]}) finished"
        else
            echo "ERR GPU ${AGENT_GPU[$i]} (routes ${AGENT_SUBSET[$i]}) exited $code — see ${LOG_DIR}/agent_gpu${AGENT_GPU[$i]}.log"
            failures=$((failures + 1))
        fi
    done
    [ "$failures" -gt 0 ] && echo "run ${RUN_IDX}: $failures GPU(s) failed — inspect logs; re-launch to resume."

    # Aggregate this run's shards.
    local merged="${RUN_DIR}/results_merged.json"
    echo "Aggregate this run with:"
    echo "  ${PYBIN} $AGGREGATOR --checkpoints-dir ${CKPT_DIR} -e ${merged}"
    if [ "$AUTO_AGGREGATE" = "1" ]; then
        echo "=== [run ${RUN_IDX}] aggregating shards ==="
        "$PYBIN" "$AGGREGATOR" --checkpoints-dir "${CKPT_DIR}" -e "${merged}" || \
            echo "Aggregation failed (is the ${CONDA_ENV} conda env active?); run the command above manually."
    fi
    return 0
}

# --- Main: repeat the eval NUM_RUNS times into run_1..run_N ---
for ((RUN_IDX = 1; RUN_IDX <= NUM_RUNS; RUN_IDX++)); do
    RUN_DIR="${RESULTS_DIR}/run_${RUN_IDX}"
    LOG_DIR="${RUN_DIR}/logs"
    CKPT_DIR="${RUN_DIR}/checkpoints"
    SAVE_DIR="${RUN_DIR}/agent_saves"

    echo ""
    echo "##########################################################"
    echo "#  RUN ${RUN_IDX}/${NUM_RUNS}  ->  ${RUN_DIR}"
    echo "##########################################################"

    if [ "$SKIP_COMPLETE" = "1" ] && run_is_complete "$RUN_DIR"; then
        echo "run ${RUN_IDX} already complete (results_merged.json has a global_record) — skipping."
        continue
    fi

    mkdir -p "$LOG_DIR" "$CKPT_DIR" "$SAVE_DIR"
    run_one || echo "run ${RUN_IDX}: CARLA startup failed — moving on; re-launch to retry this run."
    kill_carla_servers

    if [ "$RUN_IDX" -lt "$NUM_RUNS" ]; then
        echo "Waiting ${RUN_STAGGER}s before the next run..."
        sleep "$RUN_STAGGER"
    fi
done

echo ""
echo "=========================================="
echo "All ${NUM_RUNS} run(s) done for '${EXPERIMENT_NAME}'. Per-run summaries:"
for ((RUN_IDX = 1; RUN_IDX <= NUM_RUNS; RUN_IDX++)); do
    m="${RESULTS_DIR}/run_${RUN_IDX}/results_merged.json"
    if run_is_complete "${RESULTS_DIR}/run_${RUN_IDX}"; then
        echo "  run_${RUN_IDX}: complete -> $m"
    else
        echo "  run_${RUN_IDX}: INCOMPLETE (re-launch to resume)"
    fi
done
echo ""
echo "Average across runs with carla_garage/leaderboard/testbench.ipynb (set RESULTS_ROOTS to the"
echo "run_*/checkpoints dirs under ${RESULTS_DIR})."
echo "=========================================="
