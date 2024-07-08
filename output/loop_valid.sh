#!/bin/bash

cd Splitting/160-20-20/valid_1JPS_split_160-20-20/
for i in output_renet_*
do
	python valid_result.py $i 80
	mv Scores scores_$i
	echo $i >> metrics_valid
	cat scores_$i >> metrics_valid
done

cd ../../100-50-50/valid_1JPS_split_100-50-50/
for i in output_renet_*
do
	python valid_result.py $i 80
	mv Scores scores_$i
	echo $i >> metrics_valid
	cat scores_$i >> metrics_valid
done

cd ../../50-75-75/valid_1JPS_split_50-75-75/
for i in output_renet_*
do
	python valid_result.py $i 80
	mv Scores scores_$i
	echo $i >> metrics_valid
	cat scores_$i >> metrics_valid
done
