python -m torch.distributed.launch $@ train_dist_multi.py \
--config configs/pt/huabu_dist-webvid2m-pt-i2k.json # --launcher pytorch