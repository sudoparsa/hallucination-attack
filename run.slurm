#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH -A punim2198 


module load GCCcore/11.3.0
module load Python/3.11.3

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

source ../UMDCollab/virtualenv/Python3.11.3/bin/activate


python train_projector.py --online_training --cache_path ../UMDCollab/cache --coco_dir ../UMDCollab/COCO
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

