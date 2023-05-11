#!/usr/bin bash
det_methods = ('ADB' 'DeepUnk' 'DTGNS')
dis_methods = ('MORE' 'DeepAligned' 'DTGNS')
# det_methods=('DTGNS')
# dis_methods=('DTGNS')
datanames=('semeval' 'wiki80' 'wiki20m' 'nyt10m')
for s in 0 1 2 3 4
do
    for dataname in ${datanames[@]}
    do
        for det in ${det_methods[@]}
        do
            for dis in ${dis_methods[@]}
            do
                for kcr in 0.25 0.5 0.75
                do
                    for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
                    do
                        python run.py \
                            --backbone bert \
                            --detection_method $det \
                            --discovery_method $dis \
                            --method_type semi_supervised \
                            --seed $s \
                            --dataname $dataname \
                            --task_type relation_discovery \
                            --gpu_id 1 \
                            --known_cls_ratio $kcr \
                            --train_model 0 \
                            --is_pipe 1 \
                            --labeled_ratio $labeled_ratio
                    done
                done
            done
        done
    done
done