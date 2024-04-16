#!/bin/bash
#BATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --array=0-349
#SBATCH --output=outs/part_2/HW2_%j_%a_stdout.txt
#SBATCH --error=errors/part_2/HW2_%j_%a_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=HW2_part2
#SBATCH --mail-user=vineel.palla-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504322/homeworks/HW2/

## Load the TensorFlow setup
. /home/fagg/tf_setup.sh
conda activate dnn


## Run the Python script with parameters
python hw1_base_skel_dropout.py --results_path ./results/part_2 --epochs 1000 --output_type ddtheta --predict_dim 1 --activation_out linear --exp_index $SLURM_ARRAY_TASK_ID  -vv --hidden 500 250 125 75 36 17 --cpus_per_task $SLURM_CPUS_PER_TASK
