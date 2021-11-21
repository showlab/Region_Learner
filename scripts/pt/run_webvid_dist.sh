python -m torch.distributed.launch --nproc_per_node 8 --master_port 29112 train_dist_multi.py \
--config configs/pt/dist-webvid2m-pt-i2k.json # --launcher pytorch