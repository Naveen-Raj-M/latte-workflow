#!/bin/bash
#SBATCH -J checkerboard_tests                     # job name
#SBATCH -o checkerboard_tests.o%j                 # stdout/stderr file (%j = job id)
#SBATCH -N 32                                     # number of nodes (= (# MPI ranks * # OMP Threads / # cores per node))
#SBATCH -n 248                                    # total MPI ranks (# receivers in final data); keep equal to NUM_RANK below
#SBATCH -p your_partition                         # queue / partition name
#SBATCH -t 15:00:00                               # wall time (hh:mm:ss) for ALL tests in the list
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
# CHECKERBOARD TESTS
# ==============================================================================
# Each test needs, under BASE_CHECKERBOARD_TEST_DIR:
#   inversion_runs_perturbed/<test>.rb   LATTE parameter file for the checkerboard model
#   inversion_runs/<test>.rb             LATTE parameter file for the noisy-data inversion
# and a config in configs/checkerboard_test/ (CONFIG_FILE) whose "defaults" list includes "default".
# ==============================================================================

# Test names; each must match its .rb file names, e.g. <pattern>_<amplitude>_<wavelength>_<depth>
tests=(
    "checkerboard_test_name_1"
    "checkerboard_test_name_2"
    "checkerboard_test_name_3"
)

for test in "${tests[@]}"; do

    declare -A CFG=(
      # ---------- experiment ----------
      [TEST]="${test}"

      # ---------- resources ----------
      [NUM_RANK]=248                              # MPI ranks passed to mpirun
      [OMP_NUM_PROC]=14                           # OpenMP threads per rank

      # ---------- paths ----------
      [LATTE_WORKFLOW_DIR]="/path/to/latte-workflow"
      [BASE_CHECKERBOARD_TEST_DIR]="/path/to/checkerboard_tests/grid_name/pattern_name"
      [BINDIR]="/path/to/latte_traveltime/bin"   # directory containing x_tloc3
      [CONFIG_FILE]="checkerboard_config_name"   # config name in configs/checkerboard_test/, without .yaml

      # ---------- options ----------
      [SOURCE_RECEIVER_EXCHANGE]=false            # true if LATTE used stations as virtual sources

      # ---------- step flags ----------
      [RUN_INVERSION_PERTURBED]=true              # 1. LATTE on the checkerboard model
      [RUN_ADD_NOISE]=true                        # 2. add noise to the synthetic travel times
      [RUN_INVERSION]=true                        # 3. LATTE inversion of the noisy travel times
    )

    # Serialize dictionary into a single string
    CFG_STR=""
    for k in "${!CFG[@]}"; do
        CFG_STR+="$k=${CFG[$k]};;"
    done

    echo "===================================================="
    echo " Running checkerboard test: ${CFG[TEST]}"
    echo "===================================================="

    SHELL_SCRIPT="${CFG[LATTE_WORKFLOW_DIR]}/shell_scripts/checkerboard_test_workflow.sh"

    if ! bash "${SHELL_SCRIPT}" "${CFG_STR}"; then
        echo "Test ${CFG[TEST]} FAILED — continuing to next test"
    fi

    unset CFG

done
