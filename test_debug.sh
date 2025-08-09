HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
python -u test.py \
--epoch=200 \
--LR_dir=testset/RealSR/LR \
--SR_dir=result/RealSR