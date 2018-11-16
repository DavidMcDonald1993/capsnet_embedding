#!/bin/bash

n_job=4
# rm -r ../{models,plots,logs,tensorboards,walks}/ppi/*

# perform walks
parallel -j $n_job -q python exponential_mapping_gpu.py --dataset ppi \
	 {1} -b 32 --lr 0.01 --just-walks --only-lcc --seed {2} {3} ::: \
	--no-attributes --multiply-attributes --jump-prob={0.05,0.1,0.2,0.5,1.0} ::: {0..5} :::\
	--evaluate-link-prediction --evaluate-class-prediction