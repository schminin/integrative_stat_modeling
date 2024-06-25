#!/bin/bash
#SBATCH --job-name ww_filter
#SBATCH --output log/log.%A_%a.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --time 6-10:00:00
#SBATCH --array 0-2%2

module purge

module load linux-rocky8-x86_64/gcc/9.4.0/julia/1.8.2-gsjpvfv

time /home/vincent/julia-1.10.4/bin/julia --project=/home/vincent/wastewater_inference/integrative_stat_modelling -p 10 BootstrapFilter.jl
