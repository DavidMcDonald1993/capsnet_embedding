#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --time 24:00:00
#SBATCH --mem 16gb

set -e

module purge
module load bluebear
module load apps/python2/2.7.11
module load apps/tensorflow/1.3.1-python-2.7.11
module load apps/keras/2.0.8-python-2.7.11

cd src/

python graphcaps.py -s 5 5 -f 6 4 -a 32 16 -n 7 1 -d 4 2 --nneg 10 
