#!/bin/bash

cd Splitting/50-75-75/output_1JPS_split_50-75-75/
for i in output_*
do
	python result.py $i
	mv Bubble_Heatmap.png bubble_$i.png
	mv Scatter.png scatter_$i.png
	mv Scores scores_$i
	echo $i >> metrics
	cat scores_$i >> metrics
done

cd ../../100-50-50/output_1JPS_split_100-50-50/
for i in output_*
do
	python result.py $i
	mv Bubble_Heatmap.png bubble_$i.png
	mv Scatter.png scatter_$i.png
	mv Scores scores_$i
	echo $i >> metrics
	cat scores_$i >> metrics
done

cd ../../160-20-20/output_1JPS_split_160-20-20/
for i in output_*
do
	python result.py $i
	mv Bubble_Heatmap.png bubble_$i.png
	mv Scatter.png scatter_$i.png
	mv Scores scores_$i
	echo $i >> metrics
	cat scores_$i >> metrics
done