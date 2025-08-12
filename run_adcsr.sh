#!/bin/bash

in_dir=testset/DRealSR/LR
out_dir=testset/DRealSR/model_output_HR

# in_dir=testset/DigitalZoom_SR_dataset_20250307/LR
# out_dir=testset/DigitalZoom_SR_dataset_20250307/results_fp16

# in_dir=testset/sg_cropped/LR_square
# out_dir=testset/sg_cropped/LR_square_results_fp16

mkdir -p $out_dir

#  python test.py \
 python test_fp16_vs_fp32.py \
   --LR_dir=$in_dir \
   --SR_dir=$out_dir \
   --compare_precision