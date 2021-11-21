python -m torch.distributed.launch --nproc_per_node 8 train_dist_multi.py \
--config configs/dist-cc-webvid2m-pt-i2k.json # --launcher pytorch