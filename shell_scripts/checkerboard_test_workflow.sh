#!/bin/bash

# ==============================================================================
# CHECKERBOARD TEST WORKFLOW
# ==============================================================================
# Runs the LATTE-dependent steps of one checkerboard resolution test. Called once
# per test by checkerboard_test_job_submission.slurm with a serialized config.
#
# Prerequisite (not run here): checkerboard_test.main with
#   workflow.step1_create_checkerboard=true
# has written the perturbed model binaries and the LATTE parameter files:
#   ${BASE_CHECKERBOARD_TEST_DIR}/inversion_runs_perturbed/${TEST}.rb
#   ${BASE_CHECKERBOARD_TEST_DIR}/inversion_runs/${TEST}.rb
#
# Steps (each toggled by a RUN_* flag; matches the steps the default config executes):
#   1. RUN_INVERSION_PERTURBED : LATTE on the checkerboard model, one iteration
#                                -> synthetic travel times for the perturbed model
#   2. RUN_ADD_NOISE           : checkerboard_test.main (workflow.step2_add_noise)
#                                -> noisy_traveltimes/ used as the observed data
#   3. RUN_INVERSION           : LATTE inversion of the noisy travel times
# ==============================================================================

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 \"KEY=value;;KEY=value;;...\""
  exit 1
fi

# ------------------------------
# Deserialize dictionary
# ------------------------------
declare -A CFG

IFS=';;' read -ra PAIRS <<< "$1"
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  key="${pair%%=*}"
  value="${pair#*=}"
  CFG["$key"]="$value"
done

echo "================ CONFIG ================="
for k in "${!CFG[@]}"; do
  printf "%-30s = %s\n" "$k" "${CFG[$k]}"
done
echo "========================================"

# ---------- experiment ----------
TEST="${CFG[TEST]}"

# ---------- resources ----------
NUM_RANK="${CFG[NUM_RANK]}"
OMP_NUM_PROC="${CFG[OMP_NUM_PROC]}"

# ---------- paths ----------
LATTE_WORKFLOW_DIR="${CFG[LATTE_WORKFLOW_DIR]}"
BASE_CHECKERBOARD_TEST_DIR="${CFG[BASE_CHECKERBOARD_TEST_DIR]}"
BINDIR="${CFG[BINDIR]}"
CONFIG_FILE="${CFG[CONFIG_FILE]}"

INVERSION_PERTURBED_RUNS_DIR="${BASE_CHECKERBOARD_TEST_DIR}/inversion_runs_perturbed"
INVERSION_RUNS_DIR="${BASE_CHECKERBOARD_TEST_DIR}/inversion_runs"
TEST_OUTPUT_DIR="${BASE_CHECKERBOARD_TEST_DIR}/${TEST}"

# ---------- options ----------
SOURCE_RECEIVER_EXCHANGE="${CFG[SOURCE_RECEIVER_EXCHANGE]:-false}"

# ---------- step flags ----------
RUN_INVERSION_PERTURBED="${CFG[RUN_INVERSION_PERTURBED]:-false}"
RUN_ADD_NOISE="${CFG[RUN_ADD_NOISE]:-false}"
RUN_INVERSION="${CFG[RUN_INVERSION]:-false}"

export OMP_NUM_THREADS="${OMP_NUM_PROC}"

# Run LATTE for one parameter file and write everything to a timestamped log.
#   $1 = log directory, $2 = run title, $3 = parameter file
run_latte() {
    local log_dir="$1" title="$2" param_file="$3"
    local executable="x_tloc3"

    if [[ ! -f "${param_file}" ]]; then
        echo "ERROR: Parameter file not found: ${param_file}"
        exit 1
    fi

    mkdir -p "${log_dir}"
    local log_file="${log_dir}/run_${TEST}_simulation_log_$(date +"%Y-%m-%d_%H-%M-%S").txt"

    {
    echo "========================================================"
    echo "${title}"
    echo "Started at: $(date)"
    echo "MPI ranks      : ${NUM_RANK}"
    echo "OMP threads    : ${OMP_NUM_THREADS}"
    echo "Executable     : ${executable}"
    echo "Parameter file : ${param_file}"
    echo "========================================================"
    echo
    echo "================ PARAMETER FILE ========================"
    cat "${param_file}"
    echo "========================================================"
    echo
    echo "Running command:"
    echo "mpirun -np ${NUM_RANK} ${BINDIR}/${executable} ${param_file}"
    echo

    mpirun -np "${NUM_RANK}" "${BINDIR}/${executable}" "${param_file}"

    echo
    echo "Finished at: $(date)"
    echo "========================================================"
    } > "${log_file}" 2>&1

    echo "Log: ${log_file}"
}

# Run one checkerboard_test.main step; every other workflow step is switched off.
#   $1 = step name (currently only step2_add_noise)
#   remaining arguments = additional Hydra overrides
run_checkerboard_step() {
    local step="$1"; shift
    local cmd=(
        python3 -m checkerboard_test.main --config-name "${CONFIG_FILE}"
        "general.output_dir=${TEST_OUTPUT_DIR}"
        "workflow.step1_create_checkerboard=false"
        "workflow.step2_add_noise=false"
        "workflow.${step}=true"
        "$@"
    )
    echo "Executing:"
    echo "${cmd[*]}"
    echo
    (cd "${LATTE_WORKFLOW_DIR}" && "${cmd[@]}")
}

echo "===================================================="
echo "Starting checkerboard test workflow for: ${TEST}"
echo "===================================================="

# ===========================================================================
# STEP 1: LATTE on the checkerboard model (synthetic travel times)
# ===========================================================================

if [[ "${RUN_INVERSION_PERTURBED}" == "true" ]]; then
    echo "Step 1: LATTE with the checkerboard velocity model (single iteration)"
    run_latte "${INVERSION_PERTURBED_RUNS_DIR}/logs" \
              "Run ${TEST}: Inversion with perturbed checkerboard model" \
              "${INVERSION_PERTURBED_RUNS_DIR}/${TEST}.rb"
    echo "Step 1 completed."
fi

# ===========================================================================
# STEP 2: Add noise to the synthetic travel times
# ===========================================================================

if [[ "${RUN_ADD_NOISE}" == "true" ]]; then
    echo "Step 2: Adding noise to synthetic travel times"

    NOISE_OVERRIDES=(
        "input.synthetic_traveltime_dir=${INVERSION_PERTURBED_RUNS_DIR}/${TEST}/iteration_0/synthetic"
        "traveltime_noise.source_receiver_exchange=${SOURCE_RECEIVER_EXCHANGE}"
    )
    if [[ "${SOURCE_RECEIVER_EXCHANGE}" == "true" ]]; then
        # LATTE's virtual-source table; input.geometry_dir (from the config) is cross-checked against it
        NOISE_OVERRIDES+=("input.exchange_geometry_file=${INVERSION_PERTURBED_RUNS_DIR}/${TEST}/exchange_geometry.txt")
    fi

    run_checkerboard_step step2_add_noise "${NOISE_OVERRIDES[@]}"
    echo "Step 2 completed."
fi

# ===========================================================================
# STEP 3: LATTE inversion of the noisy travel times
# ===========================================================================

if [[ "${RUN_INVERSION}" == "true" ]]; then
    echo "Step 3: Inversion using noisy travel times"
    run_latte "${INVERSION_RUNS_DIR}/logs" \
              "Run ${TEST}: Checkerboard inversion of noisy travel times" \
              "${INVERSION_RUNS_DIR}/${TEST}.rb"
    echo "Step 3 completed."
fi

echo "===================================================="
echo "Checkerboard test workflow finished successfully"
echo "===================================================="
