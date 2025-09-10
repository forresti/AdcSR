#!/bin/bash

in_dir=testset/sg_cropped/faces_crop_fi_512
out_dir=testset/sg_cropped/faces_crop_fi_512_results

mkdir -p $out_dir

 python export.py \
   --LR_dir=$in_dir \
   --SR_dir=$out_dir