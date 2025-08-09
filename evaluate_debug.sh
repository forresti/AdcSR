HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
python -u evaluate.py \
--HR_dir=testset/RealSR/HR \
--SR_dir=result/RealSR
