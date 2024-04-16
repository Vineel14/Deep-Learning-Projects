#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --output=outs/mha/hw6_%04a_stdout.txt
#SBATCH --error=errors/mha/hw6_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=hw6_2
#SBATCH --mail-user=vineel.palla-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504322/homeworks/HW6
#SBATCH --array=0-4
##
#################################################
## Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf
module load cuDNN/8.9.2.26-CUDA-12.2.0


python hw6_base.py @oscer.txt @exp.txt @mha.txt --exp_index $SLURM_ARRAY_TASK_ID 
