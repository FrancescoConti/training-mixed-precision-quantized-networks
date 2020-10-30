#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python3 main_binary.py \
  -a mobilenet \
  --mobilenet_width 1.0 \
  --mobilenet_input 128 \
  --dataset imagenet \
  --weight_bits 8 \
  --activ_bits 8 \
  --gpus 0 \
  -j 8 \
  --epochs 12 \
  -b 128 \
  --quantize \
  --terminal \
  --resume checkpoint/mobilenet_1.0_128_best.pth
 
