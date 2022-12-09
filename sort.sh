#!/bin/bash
#SBATCH --account hjcl4613-rfi-core
#SBATCH --qos rfi
#SBATCH --mail-user joss.whittle@rfi.ac.uk
#SBATCH --mail-type ALL
#SBATCH --time 20
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 36
#SBATCH --gpus-per-node 0
#SBATCH --output job-%j.log

module purge
module load baskerville
module load Python/3.9.5-GCCcore-10.3.0
module load OpenMPI/4.1.1-GCC-10.3.0

set -x

# Root storage directory
export PROJECT_DIR="/bask/projects/h/hjcl4613-rfi-core/ffnr0871"

# Directory for the input data
export DATA_DIR="$PROJECT_DIR/CPMU_I04/CPMU_I04"
mkdir -p $DATA_DIR

# Directory to run this job in
export JOB_WORK_DIR="$PROJECT_DIR/CPMU_I04/job-$SLURM_JOB_ID"
mkdir -p $JOB_WORK_DIR
cd $JOB_WORK_DIR

# Copy the source data into a new directory for the job
mkdir -p "$JOB_WORK_DIR/CPMU_I04"
cp -vR $DATA_DIR $JOB_WORK_DIR

# Install Opt-ID in the job directory
export OPTID_PATH="$JOB_WORK_DIR/Opt-ID"
git clone --depth 1 -b v2 https://github.com/rosalindfranklininstitute/Opt-ID.git $OPTID_PATH
export PYTHONPATH="$OPTID_PATH:$PYTHONPATH"

python3 -m venv --system-site-packages optidjobenv
source optidjobenv/bin/activate
pip3 install -r "$OPTID_PATH/requirements.txt"

# Generate the .json, .mag, and .h5  files on the worker 0
python3 -m IDSort.src.optid -v 4 CPMU_I04/config.yaml .

# Launch the sort job on all the workers with --cluster-on
python3 -m IDSort.src.optid -v 4 --sort CPMU_I04/config.yaml .
