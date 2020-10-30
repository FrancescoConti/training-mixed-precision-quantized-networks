#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python3 main_binary.py \
  -a mobilenet \
  --mobilenet_width 1.0 \
  --mobilenet_input 224 \
  --save Imagenet/mobilenet_224_1.0_mixed7_n \
  --dataset imagenet \
  --weight_bits 8 \
  --activ_bits 8 \
  --gpus 0 \
  -j 8 \
  --epochs 3 \
  -b 64 \
  --save_check \
  --quantize \
  --results results_mixed7_n \
  --initial_equalization \
  --mem_constraint [8192000,400000]

