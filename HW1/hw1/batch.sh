#!/bin/bash
#SBATCH --partition=disc_dual_a100_students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --array=0-360
#SBATCH --output=outs/r1/HW1_%j_%a_stdout.txt
#SBATCH --error=errors/r1/HW1_%j_%a_stderr.txt
#SBATCH --time=00:50:00
#SBATCH --job-name=HW1
#SBATCH --mail-user=vineel.palla-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504322/homeworks/HW1/

## Load the TensorFlow setup
. /home/fagg/tf_setup.sh
conda activate dnn


## Run the Python script with parameters
python hw1_base_skel.py --results_path ./results/r1 --epochs 150 --output_type ddtheta --predict_dim 0 --activation_out linear --exp_index $SLURM_ARRAY_TASK_ID  -vv --hidden 500 500


