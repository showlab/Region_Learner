. /data/miniconda3/etc/profile.d/conda.sh
which conda

# -----------------------------------------------------------------------------
# activate conda env
# -----------------------------------------------------------------------------
conda deactivate
conda activate env-3.6.8
PYTHON=${PYTHON:-"/data/miniconda3/envs/env-3.6.8/bin/python"}

# mirrors.tencent.com/todacc/venus-std-ext-cuda11.0-py3.6-deepspeed0.4.0-pytorch1.7-root:0.1.1
# -----------------------------------------------------------------------------
# goto workdir
# -----------------------------------------------------------------------------
echo "load path ....."
cd /cfs/cfs-4260a4096/mds5/yanrui/code/frozen_dist
# cd /cfs/cfs-4260a4096/mds5/awinywang/Code/VLP/alex_frozen_dist

nvidia-smi

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
pip install neptune-contrib --user
pip install addict future lmdb numpy Pillow pyyaml requests scikit-image scipy tb-nightly tqdm yapf
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 black==19.3b0 flake8 isort parameterized setuptools simplejson

pip install ffmpeg
# pip install pytorchvideo pandas


MSRVTT_root="/cfs/cfs-4260a4096/260a4096/417-mds5/awinywang417/Data/MSRVTT/MSRVTT/"
MSVD_root="/cfs/cfs-4260a4096/260a4096/public_datasets/MSVD/YouTubeClips/"
DiDeMo_root="/cfs/cfs-4260a4096/260a4096/public_datasets/DiDeMo/video/"
LSMDC_root="/cfs/cfs-4260a4096/260a4096/mds10/LSMDC/"

MSRVTT_save_dir="./results/WebVid/ft/MSRVTT/"
MSVD_save_dir="./results/WebVid/ft/MSVD/"
DiDeMo_save_dir="./results/WebVid/ft/DiDeMo/"
LSMDC_save_dir="./results/WebVid/ft/LSMDC/"


nproc_per_node=1

WebVid2M_1f_pti2k_mw="results/WebVid/pt/ViT/CLU/YYLR_2048/models/full-WebVid2M-1f-pti2k/1004_020216/checkpoint-epoch99.pth"


# # # # Mr Yan
# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/ft/msrvtt_4f_i21k.json --launcher pytorch \
# --save_dir $save_dir'debug/' --load_checkpoint $WebVid2M_1f_pti2k_mw \
# --data_dir_0 $MSRVTT_root --clustering true --learning_rate1 3e-5 --vis_saving --debug



# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/ft/MSVD_4f.json --launcher pytorch \
# --save_dir $MSVD_save_dir'debug/' --load_checkpoint $WebVid2M_1f_pti2k_mw \
# --data_dir_0 $MSVD_root --learning_rate1 3e-5 --debug


# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/ft/DiDeMo_4f.json --launcher pytorch \
# --save_dir $DiDeMo_save_dir'debug/' --load_checkpoint $WebVid2M_1f_pti2k_mw \
# --data_dir_0 $DiDeMo_root --learning_rate1 3e-5 --debug


# python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
# --config configs/ft/LSMDC_4f.json --launcher pytorch \
# --save_dir $LSMDC_save_dir'debug/' --load_checkpoint $WebVid2M_1f_pti2k_mw \
# --data_dir_0 $LSMDC_root --learning_rate1 3e-5 --debug


WV_1f_pti2k_mw="results/WebVid/pt/ViT/CLU_SELECT/full-WebVid2M-1f-pti2k/1019_020413/model_best.pth"
python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
--config configs/ft/msrvtt_8f_i21k.json --launcher pytorch \
--save_dir $MSRVTT_save_dir"/ViT/CLU_SELECT/full-WebVid2M-1f-pti2k/1019_020413/8f/debug" --load_checkpoint $WV_1f_pti2k_mw \
--data_dir_0 $MSRVTT_root --learning_rate1 3e-5 --schedule 101 --debug --batch_size_0 8


if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi