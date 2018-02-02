#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --time 10:00:00
#SBATCH --mem 16gb

set -e

module purge;
module load bluebear
module load apps/python2/2.7.11
module load apps/tensorflow/1.3.1-python-2.7.11
module load apps/keras/2.0.8-python-2.7.11

cd src/

python graphcaps.py -s 5 5 5 -f 8 16 32 -a 128 64 32 -n 16 7 1 -d 4 4 2 --nneg 10 
