#!/bin/bash
#SBATCH -J forward_model                          # job name
#SBATCH -o forward_model.o%j                      # stdout/stderr file (%j = job id)
#SBATCH -N 32                                     # number of nodes (= (# MPI ranks * # OMP Threads / # cores per node))
#SBATCH -n 248                                    # total MPI ranks (# receivers in final data); keep equal to NUM_RANK below
#SBATCH -p your_partition                         # queue / partition name
#SBATCH -t 3:00:00                                # wall time (hh:mm:ss)
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
# FORWARD MODELLING
# ==============================================================================
# Computes travel times (and optionally ray paths) through a velocity model.
# The parameter file (.rb) is written by hand or from the preprocess output.

# --- Configuration ---
RUN="forward_run_name"
RUN_TITLE="Run ${RUN}: Forward model of <velocity model> on a <grid size> grid"
BASE_DIR="/path/to/forward_runs"
LOG_DIR="${BASE_DIR}/logs"
PARAM_FILE="${BASE_DIR}/${RUN}.rb"
EXECUTABLE="x_eikonal3"                           # x_eikonal3 (forward only), x_fatt3 or x_tloc3
BINDIR="/path/to/latte_traveltime/bin"            # directory containing the executables

# MPI + OpenMP setup
NUM_RANK=248
export OMP_NUM_THREADS=14

if [[ ! -f "${PARAM_FILE}" ]]; then
    echo "ERROR: Parameter file ${PARAM_FILE} not found!"
    exit 1
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP_START=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/run_${RUN}_simulation_log_${TIMESTAMP_START}.txt"

echo "Starting ${RUN_TITLE} at ${TIMESTAMP_START} ..."

# --- Execution & Logging ---
{
echo "========================================================"
echo "${RUN_TITLE}"
echo "Log generated on: $(date)"
echo "========================================================"
echo ""

echo "========================================================"
echo "PARAMETER FILE CONTENTS: ${PARAM_FILE}"
echo "========================================================"
cat "${PARAM_FILE}"
echo ""
echo "========================================================"
echo ""

echo "Running command: mpirun -np ${NUM_RANK} ${BINDIR}/${EXECUTABLE} ${PARAM_FILE}"
echo ""
mpirun -np "${NUM_RANK}" "${BINDIR}/${EXECUTABLE}" "${PARAM_FILE}"
echo ""
echo "========================================================"
echo "Simulation finished at: $(date)"
echo "========================================================"
} > "${LOG_FILE}" 2>&1

echo "Simulation finished at $(date +"%Y-%m-%d_%H-%M-%S"). Output has been logged to ${LOG_FILE}"
