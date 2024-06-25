#!/bin/bash
#SBATCH --job-name synth_filter
#SBATCH --output log/log.%A_%a.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 6-10:00:00
#SBATCH --array 0-3

module purge

module load linux-rocky8-x86_64/gcc/9.4.0/julia/1.8.2-gsjpvfv

time /home/vincent/julia-1.10.4/bin/julia --project=/home/vincent/nils_vincent_colab -p 8 ParticleFilterRun.jl
