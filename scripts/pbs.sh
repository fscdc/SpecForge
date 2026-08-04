#!/bin/bash
#PBS -P CFP04-CF-021
#PBS -j oe
#PBS -k oed
#PBS -N pytorch
#PBS -q auto
#PBS -l select=1:ngpus=4
#PBS -l walltime=96:00:00

cd $PBS_O_WORKDIR;

# /app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif"
module load singularity

singularity exec -e $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID


source ~/.bashrc
conda activate specforge

bash ./scripts/mmflash_data_hpc.sh

EOF