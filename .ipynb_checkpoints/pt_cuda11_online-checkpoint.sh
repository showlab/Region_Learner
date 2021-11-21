. /data/miniconda3/etc/profile.d/conda.sh
which conda

# -----------------------------------------------------------------------------
# activate conda env
# -----------------------------------------------------------------------------
conda deactivate
conda activate env-3.8.8
PYTHON=${PYTHON:-"/data/miniconda3/envs/env-3.8.8/bin/python"}

# mirrors.tencent.com/todacc/venus-std-ext-cuda11.0-py3.6-deepspeed0.4.0-pytorch1.7-root:0.1.1
# -----------------------------------------------------------------------------
# goto workdir
# -----------------------------------------------------------------------------
echo "load path ....."
# cd /cfs/cfs-4260a4096/260a4096/mds5/yanrui/code/frozen_dist


# export NCCL_DEBUG=INFO
# pip install neptune-contrib
pip install decord
pip install timm==0.4.5
pip install dominate
pip install sacred
pip install numpy nltk gensim textblob googletrans
pip install textaugment
pip install gensim==3.4.0
pip install addict future lmdb numpy Pillow pyyaml requests scikit-image scipy tb-nightly tqdm yapf timm==0.3.2
pip install av
pip install psutil
pip install msgpack
pip install humanize
pip install ipdb
pip install scipy
pip install sklearn
pip install transformers
pip install timm==0.4.5
pip install einops
pip install numpy nltk gensim textblob googletrans
# pip install neptune-contrib --user
pip install addict future lmdb numpy Pillow pyyaml requests scikit-image scipy tb-nightly tqdm yapf
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 black==19.3b0 flake8 isort parameterized setuptools simplejson
pip install pytorchvideo pandas
# pip install pandas


# # Mr Yan
# # python -m torch.distributed.launch $@ train_dist_multi.py --config configs/pt/dist-webvid2m-pt-i2k.json --launcher pytorch
# # python -m torch.distributed.launch --nproc_per_node 4 $@ train_dist_multi.py --config configs/pt/dist-webvid2m-pt-i2k.json --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 21302 train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k.json --launcher pytorch --frist_num_workers 8 #--model 'resnet18'

WebVid_root="/cfs/cfs-4260a4096/260a4096/445-mds11/WebVid/"
CC3M_root="/cfs/cfs-4260a4096/260a4096/445-mds11/CC3M/"


save_dir="./results/WebVid/pt/"
nproc_per_node=1

cd /cfs/cfs-4260a4096/260a4096/mds5/yanrui/code/frozen_dist
export PYTHONPATH=.:$PYTHONPATH

# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k.json --launcher pytorch \
# --num_workers_0 16 --save_dir $save_dir'debug/' --data_dir_0 $WebVid_root \
# --clustering true --batch_size_0 32 --vis_saving --debug

# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k.json --launcher pytorch \
# --num_workers_0 8 --save_dir $save_dir'ViT/CLU/YYLR_2048_ATT/' --data_dir_0 $WebVid_root \
# --clustering true --att true --not_plus true --debug --batch_size_0 12



# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k-4f.json --launcher pytorch \
# --num_workers_0 64 --save_dir $save_dir'ViT/STD_pool/YYLR/debug' --data_dir_0 $WebVid_root \
# --temporal_type 'late_fusion_pool' --debug --batch_size_0 24

# --temporal_type 'att' --debug --num_frames_0 2



# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k-4f.json --launcher pytorch \
# --num_workers_0 64 --save_dir $save_dir'MViT/STD/CLS/YYLR/debug' --data_dir_0 $WebVid_root \
# --debug --batch_size_0 32 --model 'MViT'

# CUDA_LAUNCH_BLOCKING=1
# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k.json --launcher pytorch \
# --num_workers_0 16 --save_dir $save_dir'ViT/CLU_SELECT/YYLR_2048/debug' --data_dir_0 $WebVid_root \
# --clustering true --CLU_selecting true --CLU_selecting_start 2 --debug

# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k-4f.json --launcher pytorch \
# --num_workers_0 16 --save_dir $save_dir'ViT/CLU_Tube/debug' --data_dir_0 $WebVid_root \
# --clustering true --CLU_build_tube true --temporal_type 'late_fusion_pool' --debug --batch_size_0 24

# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/pt/dist-webvid2m-pt-i2k.json --launcher pytorch \
# --num_workers_0 16 --save_dir $save_dir'ViT/CLU_Dual/debug' --data_dir_0 $WebVid_root \
# --clustering true --debug

python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
--config configs/pt/dist-webvid2m-pt-i2k-4f.json --launcher pytorch \
--num_workers_0 64 --save_dir $save_dir'ViT_4f/late_fusion_pool/CLU_TB/debug' --data_dir_0 $WebVid_root \
--temporal_type 'late_fusion_pool' --batch_size_0 24 --clustering true --CLU_build_tube true --debug

# alex
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 29112 train_dist_multi.py --config configs/pt/dist-webvid2m-pt-wtags-3-cl-loss-2-stream-i2k.json --launcher pytorch

if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi