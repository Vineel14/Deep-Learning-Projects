#!/bin/bash
#
#SBATCH --partition=disc_dual_a100_students
#SBATCH --cpus-per-task=64
#SBATCH --mem=40G
#SBATCH --output=results/outs/shallow/hw3_%j_stdout.txt
#SBATCH --error=results/errors/shallow/hw3_%j_stderr.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=hw3_shallow
#SBATCH --mail-user=vineel.palla-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504322/homeworks/HW3
#SBATCH --array=0-4

#################################################
## Do not change this line unless you have your own python/tensorflow/keras set up


. /home/fagg/tf_setup.sh
conda activate dnn_2024_02
module load cuDNN/8.9.2.26-CUDA-12.2.0


python hw3_base.py -v @exp_deep.txt @oscer.txt @net_deep.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --precache datasets_by_fold_4_objects --dataset /scratch/fagg/core50 
