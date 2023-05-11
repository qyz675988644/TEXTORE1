#!/usr/bin bash

datanames=('semeval' 'wiki80' 'wiki20m' 'nyt10m')
for s in 0
do
    for dataname in ${datanames[@]}
    do
        for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
        do
            python run.py \
                --backbone bert \
                --method ODC \
                --method_type unsupervised \
                --seed $s \
                --lr 0.1 \
                --dataname $dataname \
                --task_type relation_discovery \
                --max_length 120 \
                --optim sgd \
                --gpu_id 1 \
                --known_cls_ratio 0.25 \
                --freeze_bert_parameters 1 \
                --train_model 0
        done
    done
done