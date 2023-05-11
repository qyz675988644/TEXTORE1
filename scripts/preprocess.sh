#!/usr/bin bash

datanames=('semeval')
for s in 0
do
    for dataname in ${datanames[@]}
    do
        for kcr in 0.0 0.25 0.5 0.75 1.0
        do
            for labeled_ratio in 0.0 0.2 0.4 0.6 0.8 1.0
            do
                python preprocess.py \
                    --backbone bert \
                    --seed $s \
                    --dataname $dataname \
                    --max_length 120 \
                    --known_cls_ratio $kcr \
                    --labeled_ratio $labeled_ratio
            done       
        done
    done
done