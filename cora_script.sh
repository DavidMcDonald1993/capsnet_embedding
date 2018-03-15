#!/bin/bash
#SBATCH --qos bbgpu
#SBATCH --ntasks 1
#SBATCH --time 10-00:00:00
#SBATCH --mem 16gb
#SBATCH --array 10, 20, 50, 100

set -e

module purge
module load bluebear
module load apps/python3/3.5.2
module load apps/cuda/8.0.44
module load apps/cudnn/6.0
-I${CUDNN_ROOT}/include/cudnn.h -L${CUDNN_ROOT}/lib64/libcudnn.so
module load apps/tensorflow/1.3.1-python-3.5.2-cuda-8.0.44
module load apps/keras/2.0.8-python-3.5.2-cuda-8.0.44
module load apps/h5py/2.7.0-python-3.5.2

cd src/

python graphcaps.py --dataset cora --dim ${SLURM_ARRAY_TASK_ID}
