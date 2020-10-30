#!/bin/bash
# for mobilenet_width in 0.25 0.5 0.75 1.0; do
#  for mobilenet_input in 128 160 192 224; do
mobilenet_width=1.0
mobilenet_input=128
    echo "Export MobileNet ${mobilenet_width} ${mobilenet_input}"
    CUDA_VISIBLE_DEVICES=7 python3 main_binary.py \
    -a mobilenet \
    --mobilenet_width $mobilenet_width \
    --mobilenet_input $mobilenet_input \
    --dataset imagenet \
    --initial_equalization \
    --weight_bits 7 \
    --activ_bits 8 \
    --gpus 0 \
    -j 8 \
    --epochs 12 \
    -b 64 \
    --quantize \
    --terminal \

#    >& export_log_${mobilenet_width}_${mobilenet_input}.txt 
#   done
# done
