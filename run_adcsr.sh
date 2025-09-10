#!/bin/bash

# in_dir=testset/DRealSR/LR
# out_dir=testset/DRealSR/model_output_HR

# in_dir=testset/DigitalZoom_SR_dataset_20250307/LR
# out_dir=testset/DigitalZoom_SR_dataset_20250307/results

# in_dir=testset/sg_cropped/LR
# out_dir=testset/sg_cropped/results

# in_dir=testset/sg_cropped/LR_square
# out_dir=testset/sg_cropped/LR_square_results

# in_dir=testset/sg_cropped/faces_crop_fi_512
# out_dir=testset/sg_cropped/faces_crop_fi_512_results

in_dir=testset/sg_cropped/faces_crop_fi_256
out_dir=testset/sg_cropped/faces_crop_fi_256_results

mkdir -p $out_dir

 python test.py \
   --LR_dir=$in_dir \
   --SR_dir=$out_dir