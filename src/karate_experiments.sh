#!/bin/bash

# rm -r ../{models,plots,logs,tensorboards,walks}/karate/*

for seed in {1..5};
do
	echo "seed" $seed
	parallel -j 8 -q python exponential_mapping_gpu.py --seed $seed --dataset karate --dim {1} \
		{2} {3} -b 128 --lr 0.01 ::: {2,3,5} ::: {"--sigmoid","--softmax"} "-r "{1,3,5}" -t "{1,3,5} ::: \
		{"--no-attributes","--multiply-attributes","--add-attributes"} "--jump-prob "{0.05,0.1,0.2,0.5,1.0}
	# for dim in {2,3,5};
	# do
	# 	python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --sigmoid
	# 	python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --sigmoid --multiply-attributes
	# 	python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --sigmoid --add-attributes

	# 	for p in {0.05,0.1,0.2,0.5,1.0};
	# 	do
	# 		python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --sigmoid --jump-prob $p
	# 	done

	# 	python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --softmax
	# 	python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --softmax --multiply-attributes
	# 	python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --softmax --add-attributes

	# 	for p in {0.05,0.1,0.2,0.5,1.0};
	# 	do
	# 		python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim --softmax --jump-prob $p
	# 	done
		
	# 	for r in {1,3,5,10};
	# 	do
	# 		for t in {1,3,5,10};
	# 		do

	# 			python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim -r $r -t $t
	# 			python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim -r $r -t $t --multiply-attributes
	# 			python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim -r $r -t $t --add-attributes
	# 			for p in {0.05,0.1,0.2,0.5,1.0};
	# 			do
	# 				python exponential_mapping_gpu.py --seed $seed --dataset karate --dim $dim -r $r -t $t --jump-prob $p
	# 			done

	# 		done


	# 	done
		
	# done
done
