HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
nohup torchrun \
--nproc_per_node=1 \
--master_port=23333 \
train.py \
--batch_size=1 > g0.txt 2>&1 &