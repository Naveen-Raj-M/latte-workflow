#!/bin/bash

# ==============================================================================
# LATTE INVERSION POSTPROCESSING AND ANALYSIS
# ==============================================================================
# For every selected inversion run and iteration this script:
#   1. postprocess.main : converts LATTE binaries to CSV
#        - velocity    : updated_vp.bin / updated_vs.bin  -> velocity_model_final.csv
#        - traveltime  : synthetic/*.bin + observed picks  -> synthetic_traveltimes.csv
#        - source      : updated_sx/sy/sz/st0.bin          -> source_relocation.csv
#   2. analysis.main    : plots and statistics from those CSVs
#        - travel time residual analysis
#        - velocity model comparison (reference model vs. inverted model)
#        - source relocation analysis
#
# Both stages use the default configs (configs/postprocess/default.yaml and
# configs/analysis/default.yaml); everything below is passed as a Hydra override.
#
# Expected inversion layout (written by LATTE):
#   ${BASE_INVERSION_DIR}/${RUN_NAME}/iteration_<N>/{model,synthetic}/
#   ${BASE_INVERSION_DIR}/${RUN_NAME}/exchange_geometry.txt   (source-receiver exchange only)
#
# Output layout:
#   ${BASE_ANALYSIS_DIR}/${RUN_NAME}/iteration_<N>/
#
# Usage:
#   1. Edit the CONFIGURATION section.
#   2. bash process_inversion.sh
# ==============================================================================

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# ---------- environment ----------
LATTE_WORKFLOW_DIR="/path/to/latte-workflow"          # repository root; python modules are run from here
VENV_ACTIVATE=""                                      # e.g. "/path/to/venv/bin/activate"; leave empty if already active

# ---------- inversion runs ----------
# Each name is a directory under BASE_INVERSION_DIR (and BASE_ANALYSIS_DIR).
RUN_NAMES=(
    "inversion_run_name"
    # "another_inversion_run_name"
)

BASE_INVERSION_DIR="/path/to/inversion_runs"          # LATTE output: <run>/iteration_<N>/
BASE_ANALYSIS_DIR="/path/to/inversion_analysis"       # results are written to <run>/iteration_<N>/

# ---------- iteration selection ----------
# range : every iteration from START_ITERATION to END_ITERATION
# step  : every ITERATION_STEP-th iteration from START_ITERATION to END_ITERATION
# list  : only the iterations in ITERATION_LIST
ITERATION_MODE="range"                                # range | step | list
START_ITERATION=1
END_ITERATION=30
ITERATION_STEP=10                                     # used by "step" mode
ITERATION_LIST=(1 10 20 30)                           # used by "list" mode

# ---------- static input files (identical for every iteration) ----------
# Velocity model LATTE started from (CSV or HDF5 written by the preprocess module).
# Its grid metadata is used to read the LATTE binaries and it is the reference
# model in the velocity analysis.
REFERENCE_VELOCITY_FILE="/path/to/preprocess/output/velocity_model_init.csv"

# Observed travel times. For real data use the preprocess ground truth CSV; for
# checkerboard / spike tests use the noisy travel time CSV written by the test module.
GROUND_TRUTH_CSV="/path/to/preprocess/output/geometry_csv/geometry_ground_truth_traveltimes.csv"

SHOT_MAP_FILE="/path/to/preprocess/output/geometry/shot_to_source_id_map.txt"   # optional

# ---------- source-receiver exchange ----------
# Set true if LATTE was run with stations as virtual sources. The exchange is
# rebuilt from the geometry files; the exchange_geometry.txt LATTE wrote in each
# run directory is used as a cross-check.
SOURCE_RECEIVER_EXCHANGE=true
GEOMETRY_DIR="/path/to/preprocess/output/geometry"    # geometry.txt and shot_<n>_geometry.txt
COORDINATE_TOLERANCE=1.0                              # m, geometry receivers vs. ground truth stations
FILTERED_SHOT_IDS="[]"                                # shots dropped by LATTE, e.g. "[60,61]"
FILTERED_RECEIVER_IDS="[]"                            # stations dropped by LATTE, e.g. "[TX.ABC]"

# ---------- postprocess switches ----------
DO_VELOCITY_PROCESSING=true
DO_TRAVELTIME_PROCESSING=true
DO_SOURCE_PROCESSING=true                             # needs travel time processing (it reuses its CSV)

# ---------- postprocess options ----------
PHASE_TYPE="both"                                     # both | p | s
INTERPOLATION_MODE="sample_at_coarse"                 # sample_at_coarse | expand_to_fine
SOURCE_PARAMS_TO_PROCESS="[sx,sy,sz,st0]"             # subset of sx, sy, sz, st0

# ---------- analysis switches ----------
DO_TRAVELTIME_ANALYSIS=true
DO_VELOCITY_ANALYSIS=true
DO_SOURCE_ANALYSIS=true

# ---------- travel time analysis options ----------
MIN_RESIDUAL_THRESHOLD=-50                            # s, residuals below this are treated as unrealistic
MAX_RESIDUAL_THRESHOLD=50                             # s, residuals above this are treated as unrealistic
OUTLIER_THRESHOLD_SIGMA=3.0                           # residuals beyond N sigma are flagged as outliers

# ---------- velocity analysis options ----------
VELOCITY_MODEL1_NAME="Initial Model"                  # REFERENCE_VELOCITY_FILE
VELOCITY_MODEL2_NAME="Inverted Model"                 # LATTE result of the iteration

# ---------- source analysis options ----------
SOURCE_DEPTH_BINS=10
SOURCE_MIN_EVENTS_PER_BIN=5

# ==============================================================================
# FUNCTIONS
# ==============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# Run a command given as separate arguments, echoing it first.
run_cmd() {
    log_info "Executing: $*"
    "$@"
}

# ==============================================================================
# SETUP
# ==============================================================================

if [[ -n "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
fi

cd "${LATTE_WORKFLOW_DIR}"

case "${ITERATION_MODE}" in
    range) ITERATIONS=($(seq "${START_ITERATION}" "${END_ITERATION}")) ;;
    step)  ITERATIONS=($(seq "${START_ITERATION}" "${ITERATION_STEP}" "${END_ITERATION}")) ;;
    list)  ITERATIONS=("${ITERATION_LIST[@]}") ;;
    *)
        log_error "Unknown ITERATION_MODE: ${ITERATION_MODE} (expected range, step or list)"
        exit 1
        ;;
esac

if [[ "${DO_SOURCE_PROCESSING}" == true && "${DO_TRAVELTIME_PROCESSING}" != true ]]; then
    log_info "Note: source processing reads the travel time CSV; it must already exist from an earlier run"
fi

log_info "Runs       : ${RUN_NAMES[*]}"
log_info "Iterations : ${ITERATIONS[*]}"

# ==============================================================================
# MAIN LOOP
# ==============================================================================

FAILED=()

for RUN_NAME in "${RUN_NAMES[@]}"; do
    for ITERATION in "${ITERATIONS[@]}"; do

        TAG="${RUN_NAME}/iteration_${ITERATION}"
        ITERATION_DIR="${BASE_INVERSION_DIR}/${RUN_NAME}/iteration_${ITERATION}"
        OUTPUT_DIR="${BASE_ANALYSIS_DIR}/${RUN_NAME}/iteration_${ITERATION}"

        # Converted CSVs (names set by the postprocess defaults, passed explicitly so analysis reads the same files)
        VELOCITY_CSV="${OUTPUT_DIR}/velocity_model_final.csv"
        TRAVELTIME_CSV="${OUTPUT_DIR}/synthetic_traveltimes.csv"
        SOURCE_CSV="${OUTPUT_DIR}/source_relocation.csv"

        log_info "================================================"
        log_info "Processing ${TAG}"
        log_info "================================================"

        if [[ ! -d "${ITERATION_DIR}" ]]; then
            log_error "Iteration directory not found: ${ITERATION_DIR} (skipping)"
            FAILED+=("${TAG}: missing iteration directory")
            continue
        fi

        mkdir -p "${OUTPUT_DIR}"

        # ----------------------------------------------------------------------
        # POSTPROCESS
        # ----------------------------------------------------------------------
        # data.base_dir is the iteration directory; the model/*.bin and synthetic/
        # locations are derived from it by the config defaults.

        if [[ "${DO_VELOCITY_PROCESSING}" == true || "${DO_TRAVELTIME_PROCESSING}" == true || "${DO_SOURCE_PROCESSING}" == true ]]; then

            POSTPROCESS_CMD=(
                python3 -m postprocess.main
                "data.base_dir=${ITERATION_DIR}"
                "data.analysis_output_dir=${OUTPUT_DIR}"
            )

            if [[ "${DO_VELOCITY_PROCESSING}" == true ]]; then
                POSTPROCESS_CMD+=(
                    "velocity_processing.enabled=true"
                    "velocity_processing.phase_type=${PHASE_TYPE}"
                    "velocity_processing.reference_velocity_file=${REFERENCE_VELOCITY_FILE}"
                    "velocity_processing.interpolation_mode=${INTERPOLATION_MODE}"
                    "velocity_processing.velocity_output_csv=${VELOCITY_CSV}"
                )
            else
                POSTPROCESS_CMD+=("velocity_processing.enabled=false")
            fi

            if [[ "${DO_TRAVELTIME_PROCESSING}" == true ]]; then
                POSTPROCESS_CMD+=(
                    "travel_time_processing.enabled=true"
                    "travel_time_processing.phase_type=${PHASE_TYPE}"
                    "travel_time_processing.ground_truth_csv=${GROUND_TRUTH_CSV}"
                    "travel_time_processing.shot_map_file=${SHOT_MAP_FILE}"
                    "travel_time_processing.synthetic_tt_output_csv=${TRAVELTIME_CSV}"
                    "travel_time_processing.source_receiver_exchange=${SOURCE_RECEIVER_EXCHANGE}"
                    "travel_time_processing.filtered_shot_ids=${FILTERED_SHOT_IDS}"
                    "travel_time_processing.filtered_receiver_ids=${FILTERED_RECEIVER_IDS}"
                    "travel_time_processing.source_correction=${SOURCE_CORRECTION}"
                )
                if [[ "${SOURCE_RECEIVER_EXCHANGE}" == true ]]; then
                    POSTPROCESS_CMD+=(
                        "travel_time_processing.geometry_dir=${GEOMETRY_DIR}"
                        "travel_time_processing.exchange_geometry_file=${BASE_INVERSION_DIR}/${RUN_NAME}/exchange_geometry.txt"
                        "travel_time_processing.coordinate_tolerance=${COORDINATE_TOLERANCE}"
                    )
                fi
                if [[ "${SOURCE_CORRECTION}" == true ]]; then
                    PREVIOUS_ITERATION=$((ITERATION - 1))
                    POSTPROCESS_CMD+=(
                        "travel_time_processing.source_correction_file=${BASE_INVERSION_DIR}/${RUN_NAME}/iteration_${PREVIOUS_ITERATION}/model/updated_st0.bin"
                    )
                fi
            else
                POSTPROCESS_CMD+=("travel_time_processing.enabled=false")
            fi

            if [[ "${DO_SOURCE_PROCESSING}" == true ]]; then
                # initial_geometry_csv and filtered_shot_ids default to the travel_time_processing values
                POSTPROCESS_CMD+=(
                    "source_processing.enabled=true"
                    "source_processing.parameters_to_process=${SOURCE_PARAMS_TO_PROCESS}"
                    "source_processing.output_csv=${SOURCE_CSV}"
                )
                # The source step reads the travel time CSV even when it is not regenerated
                POSTPROCESS_CMD+=("travel_time_processing.synthetic_tt_output_csv=${TRAVELTIME_CSV}")
            else
                POSTPROCESS_CMD+=("source_processing.enabled=false")
            fi

            # Only the three conversions above are run here
            POSTPROCESS_CMD+=("ray_path_processing.enabled=false")

            if ! run_cmd "${POSTPROCESS_CMD[@]}"; then
                log_error "Postprocessing failed for ${TAG} (skipping analysis)"
                FAILED+=("${TAG}: postprocess")
                continue
            fi
            log_info "Postprocessing completed for ${TAG}"
        fi

        # ----------------------------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------------------------
        # Other analysis modules (distance, DWS, ray density, ...) are disabled by default.

        if [[ "${DO_TRAVELTIME_ANALYSIS}" == true || "${DO_VELOCITY_ANALYSIS}" == true || "${DO_SOURCE_ANALYSIS}" == true ]]; then

            ANALYSIS_CMD=(python3 -m analysis.main --config-name default)

            if [[ "${DO_TRAVELTIME_ANALYSIS}" == true ]]; then
                ANALYSIS_CMD+=(
                    "travel_time_analysis.enabled=true"
                    "travel_time_analysis.comparison_mode=ground_truth"
                    "travel_time_analysis.input_csv=${TRAVELTIME_CSV}"
                    "travel_time_analysis.output_dir=${OUTPUT_DIR}/traveltime_analysis"
                    "travel_time_analysis.min_residual_threshold=${MIN_RESIDUAL_THRESHOLD}"
                    "travel_time_analysis.max_residual_threshold=${MAX_RESIDUAL_THRESHOLD}"
                    "travel_time_analysis.outlier_threshold_sigma=${OUTLIER_THRESHOLD_SIGMA}"
                )
            fi

            if [[ "${DO_VELOCITY_ANALYSIS}" == true ]]; then
                ANALYSIS_CMD+=(
                    "velocity_analysis.enabled=true"
                    "velocity_analysis.comparison_mode=initial_vs_final"
                    "velocity_analysis.model1_csv=${REFERENCE_VELOCITY_FILE}"
                    "velocity_analysis.model2_csv=${VELOCITY_CSV}"
                    "velocity_analysis.initial_velocity_model_csv=${REFERENCE_VELOCITY_FILE}"
                    "velocity_analysis.output_dir=${OUTPUT_DIR}/velocity_analysis"
                    "velocity_analysis.model1_name=${VELOCITY_MODEL1_NAME}"
                    "velocity_analysis.model2_name=${VELOCITY_MODEL2_NAME}"
                    "velocity_analysis.interpolation_method=${VELOCITY_INTERPOLATION_METHOD}"
                    "velocity_analysis.grid_resolution=${VELOCITY_GRID_RESOLUTION}"
                )
            fi

            if [[ "${DO_SOURCE_ANALYSIS}" == true ]]; then
                ANALYSIS_CMD+=(
                    "source_analysis.enabled=true"
                    "source_analysis.comparison_mode=ground_truth"
                    "source_analysis.input_csv=${SOURCE_CSV}"
                    "source_analysis.output_dir=${OUTPUT_DIR}/source_analysis"
                    "source_analysis.parameters_to_process=${SOURCE_PARAMS_TO_PROCESS}"
                    "source_analysis.depth_bins=${SOURCE_DEPTH_BINS}"
                    "source_analysis.min_events_per_bin=${SOURCE_MIN_EVENTS_PER_BIN}"
                )
            fi

            if ! run_cmd "${ANALYSIS_CMD[@]}"; then
                log_error "Analysis failed for ${TAG}"
                FAILED+=("${TAG}: analysis")
                continue
            fi
            log_info "Analysis completed for ${TAG}"
        fi

    done
done

# ==============================================================================
# SUMMARY
# ==============================================================================

log_info "================================================"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    log_error "Finished with ${#FAILED[@]} failure(s):"
    for item in "${FAILED[@]}"; do
        log_error "  - ${item}"
    done
    exit 1
fi
log_info "All runs and iterations processed successfully"
log_info "================================================"
