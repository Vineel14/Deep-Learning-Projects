#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
## memory in MB
#SBATCH --mem=15000
#SBATCH --output=outs/rnn/run5/hw5_%04a_stdout.txt
#SBATCH --error=errors/rnn/hw5_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=hw5_2
#SBATCH --mail-user=vineel.palla-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504322/homeworks/HW5
#SBATCH --array=0-4
##
#################################################
## Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn_2024_02
module load cuDNN/8.9.2.26-CUDA-12.2.0


python hw5_base.py @oscer.txt @exp.txt @rnn_pool.txt --exp_index $SLURM_ARRAY_TASK_ID 
