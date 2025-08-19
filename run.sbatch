#!/bin/bash

#SBATCH --job-name=job
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

#SBATCH --account=....
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --exclusive





ulimit -s unlimited
module load PrgEnv-gnu
module load cray-fftw
module load mkl

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK




# Paths
WORK_DIR=$(pwd)
PROGRESS_FILE="${WORK_DIR}/progress_A.txt"
CP2K_CMD="/work/e05/e05/ash141/cp2k/cp2k/cp2k-2024.1/exe/ARCHER2/cp2k.psmp"

# Define lists for volumes, strains, and jiggles
strain_list=("strain1" "strain2" "strain3" "strain4" "strain5")
jiggle_list=("jiggle1" "jiggle2" "jiggle3" "jiggle4" "jiggle5")
volume_list=("vol-10" "vol-8" "vol-6" "vol-4" "vol-2" "vol0")

number_of_nodes=$SLURM_NNODES
num_worker_nodes=$((number_of_nodes - 1))
echo "${num_worker_nodes}"

cd $WORK_DIR



# Reset running jobs in the progress file to "n" if relaunching
if [ -f "${PROGRESS_FILE}" ]; then
    echo "Resetting running jobs to not started in the progress file..."
    sed -i 's/ r / n /g' "${PROGRESS_FILE}"
fi

# Function to create the progress file if it doesn't exist
create_progress_file() {
    if [ ! -f "${PROGRESS_FILE}" ]; then
        echo "Creating progress file..."
        {
            for volume in "${volume_list[@]}"; do
                for strain in "${strain_list[@]}"; do
                    for jiggle in "${jiggle_list[@]}"; do
                        echo "| ${volume} | ${strain} | ${jiggle} | n |"
                    done
                done
            done
        } > "${PROGRESS_FILE}"
    fi
}

# Ensure the progress file exists
create_progress_file

# Function to run a simulation and mark the progress
run_simulation() {
    local vol="$1"
    local strain="$2"
    local jiggle="$3"
    local sim_dir="${WORK_DIR}/${vol}/${strain}/${jiggle}"
    # Check if the simulation is already completed
    status=$(grep "| ${vol} | ${strain} | ${jiggle} |" "${PROGRESS_FILE}" | cut -d '|' -f 5 | xargs)
    echo "Checking status for ${vol}/${strain}/${jiggle}: Status = '$status'"
    if [ "$status" == "y" ] || [ "$status" == "r" ]; then
        echo "Skipping ${sim_dir}, already marked as completed or running."
        return
    fi
    # Path to the result.out file
    result_file="${sim_dir}/result.out"

    # Check if result.out exists
    if [[ -f "$result_file" ]]; then
        echo "File $result_file exists."

        # Check if the file contains the string "program ended at"
        if grep -q "PROGRAM ENDED AT" "$result_file"; then
            echo "Simulation completed for ${sim_dir}."
            sed -i "/| ${vol} | ${strain} | ${jiggle} |/ s/ n / y /" "${PROGRESS_FILE}"
            return
        fi
    fi

    # If not complete, mark the job as running
    sed -i "/| ${vol} | ${strain} | ${jiggle} |/ s/ n / r /" "${PROGRESS_FILE}"


    # Run the simulation
    echo "Running simulation in ${sim_dir}..."
    cd "$sim_dir"

    # Run the job in the background with exclusive access to the specified cores
    srun --hint=nomultithread --distribution=block:block ${CP2K_CMD} -i energy.inp -o result.out

    sed -i "/| ${vol} | ${strain} | ${jiggle} |/ s/ r / y /" "${PROGRESS_FILE}"
    echo "Simulation completed for ${sim_dir}."
    # Move back to the working directory
    cd $WORK_DIR
}



# Main loop to run simulations
for vol in "${volume_list[@]}"; do
    for strain in "${strain_list[@]}"; do
        for jiggle in "${jiggle_list[@]}"; do
            run_simulation "$vol" "$strain" "$jiggle"
         done
    done
done

echo "All simulations complete, exiting"


