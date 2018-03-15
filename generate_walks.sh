#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --time 3:00:00
#SBATCH --mem 16gb

set -e

module purge
module load bluebear
module load apps/python3/3.5.2
module load apps/tensorflow/1.3.1-python-3.5.2
module load apps/keras/2.0.8-python-3.5.2
module load apps/h5py/2.7.0-python-3.5.2

cd src/

python graphcaps.py --dataset CondMat