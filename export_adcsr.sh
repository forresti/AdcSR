#!/bin/bash

in_dir=testset/sg_cropped/faces_crop_fi_256
out_dir=testset/sg_cropped/faces_crop_fi_256_results

mkdir -p $out_dir

 python export.py \
   --LR_dir=$in_dir \
   --SR_dir=$out_dir \
   --fp16