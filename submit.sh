#!/bin/bash
#SBATCH --job-name=debris_generation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time 10:00:00
#SBATCH --mail-user sikka@umd.edu
#SBATCH --mail-type=ALL

#############################################################
# HPC Job Script for OpenMP C++ Debris Simulator
#############################################################

# 1. Set directories
# Update this path to where your Astrodynamics directory is on the HPC
set dir=/scratch/zt1/project/hartzell-lab/niac-project/Astrodynamics

cd $dir

# 2. Load necessary modules (adjust compiler/eigen modules based on your HPC)
# module purge
# module load gcc
# module load eigen

# 3. Clean and Compile (Optional: you can also compile manually before submitting)
make clean
make

# 4. Set OpenMP environment variables
# This tells the compiled C++ binary exactly how many cores it is allowed to use
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 5. Execute the binary
echo "Starting simulation with $OMP_NUM_THREADS threads..."
./bin/runner > ./output_debris.txt
echo "Simulation complete!"
