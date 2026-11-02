#!/bin/bash
#SBATCH -J inversion_runs                         # job name
#SBATCH -o inversion_runs.o%j                     # stdout/stderr file (%j = job id)
#SBATCH -N 32                                     # number of nodes (= (# MPI ranks * # OMP Threads / # cores per node))
#SBATCH -n 248                                    # total MPI ranks (# receivers in final data); keep equal to NUM_RANK below
#SBATCH -p your_partition                         # queue / partition name
#SBATCH -t 12:00:00                               # wall time (hh:mm:ss) for ALL runs in the list
#SBATCH -A your_allocation                        # allocation / project to charge
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=begin,end

echo "===================================================="
echo " Job started on: $(date)"
echo " Hostname(s): $(hostname)"
echo " Current working directory: $(pwd)"
echo "===================================================="

# Load aliases
echo " Sourcing ~/.bashrc ..."
source ~/.bashrc

echo " Loading Intel oneAPI and MPI modules ..."
module purge
module load intel/25.1                            # Intel compilers (ifort) used to build LATTE
module load impi                                  # Intel MPI (mpiifort, mpirun)
echo " Modules loaded:"
echo "   ifort    : $(which ifort)"
echo "   mpiifort : $(which mpiifort)"
echo "   TACC_INTEL_DIR: $TACC_INTEL_DIR"
echo "   TACC_MKL_DIR  : $TACC_MKL_DIR"
echo "   I_MPI_ROOT    : $I_MPI_ROOT"

echo " Activating latte-workflow virtual environment ..."
source /path/to/latte-workflow-venv/bin/activate
echo " Python interpreter: $(which python)"
echo " Python version: $(python --version)"

# ==============================================================================
# INVERSION RUNS
# ==============================================================================
# Runs LATTE once per parameter file in BASE_DIR. Results are written to
# <BASE_DIR>/<run>/iteration_<N>/ and can be processed with process_inversion.sh.

# --- Configuration ---
BASE_DIR="/path/to/inversion_runs"                # contains <run>.rb parameter files
LOG_DIR="${BASE_DIR}/logs"
EXECUTABLE="x_tloc3"                              # travel time + source location inversion
BINDIR="/path/to/latte_traveltime/bin"            # directory containing the executables

# Run names; each needs <BASE_DIR>/<run>.rb
RUN_NAMES=(inversion_run_name_1 inversion_run_name_2)

# MPI + OpenMP setup
NUM_RANK=248
export OMP_NUM_THREADS=14

# Create log directory
mkdir -p "$LOG_DIR"

echo "Starting batch inversion runs at $(date)"
echo "========================================================"

# Loop over all runs
for RUN in "${RUN_NAMES[@]}"; do
    
    PARAM_FILE="${BASE_DIR}/${RUN}.rb"

    RUN_TITLE="Run ${RUN}: LATTE inversion"

    TIMESTAMP_START=$(date +"%Y-%m-%d_%H-%M-%S")
    LOG_FILE="${LOG_DIR}/run_${RUN}_simulation_log_${TIMESTAMP_START}.txt"

    echo "--------------------------------------------------------"
    echo "Starting $RUN_TITLE at $TIMESTAMP_START"
    echo "Log file: $LOG_FILE"
    echo "--------------------------------------------------------"

    {
    echo "========================================================"
    echo "$RUN_TITLE"
    echo "Log generated on: $(date)"
    echo "========================================================"
    echo ""

    echo "========================================================"
    echo "PARAMETER FILE CONTENTS: $PARAM_FILE"
    echo "========================================================"
    if [ -f "$PARAM_FILE" ]; then
        cat "$PARAM_FILE"
    else
        echo "ERROR: Parameter file $PARAM_FILE not found!"
        exit 1
    fi
    echo ""
    echo "========================================================"
    echo ""

    echo "Starting TLOC simulation..."
    echo "Running command: mpirun -np ${NUM_RANK} ${BINDIR}/${EXECUTABLE} ${PARAM_FILE}"
    echo ""

    mpirun -np ${NUM_RANK} ${BINDIR}/${EXECUTABLE} "${PARAM_FILE}"

    echo ""
    echo "========================================================"
    echo "Simulation finished at: $(date)"
    echo "========================================================"

    } > "$LOG_FILE" 2>&1

    echo "Finished $RUN at $(date)"
    echo ""

done


echo "========================================================"
echo "All inversion runs completed at $(date)"
echo "========================================================"
