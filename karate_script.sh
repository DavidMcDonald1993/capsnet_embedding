#!/bin/bash
#SBATCH --qos bbgpu
#SBATCH --gres gpu:p100:1
#SBATCH --ntasks 1
#SBATCH --time 3-00:00:00
#SBATCH --mem 16G
#SBATCH --output karate.out

set -e

module purge
module load bluebear
module load apps/python3/3.5.2
module load apps/cuda/8.0.44
module load apps/cudnn/6.0
module load apps/tensorflow/1.3.1-python-3.5.2-cuda-8.0.44
module load apps/keras/2.0.8-python-3.5.2-cuda-8.0.44

cd src/

python exponential_mapping_gpu.py --dataset karate
