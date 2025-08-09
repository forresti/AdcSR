HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
--nproc_per_node=8 \
--master_port=23333 \
train.py > g0-7.txt 2>&1