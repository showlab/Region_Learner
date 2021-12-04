#  cuda11

# -----------------------------------------------------------------------------
# activate conda env
# -----------------------------------------------------------------------------
conda deactivate
conda activate env-3.6.8

# -----------------------------------------------------------------------------
# Install some important packages. (Your can also directly package them into your docker!)
# -----------------------------------------------------------------------------
pip install torch==1.8.0+cu111 torchvision==0.9.0 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install decord dominate sacred numpy nltk gensim textblob googletrans textaugment gensim==3.4.0
pip install addict future lmdb Pillow pyyaml requests scikit-image scipy tb-nightly tqdm yapf
pip install av psutil msgpack humanize ipdb scipy sklearn transformers timm==0.4.5 einops
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 black==19.3b0 flake8 isort parameterized setuptools simplejson


# -----------------------------------------------------------------------------
# goto workdir
# -----------------------------------------------------------------------------
echo "load path ....."
cd your_path/Region_Learner


# NOTE: Not all videos can be download. Maybe you need adjust the meta_data for each dataset class defined in 'data_loader'.
WebVid_root="data/WebVid/"
CC3M_root="data/CC3M/"


save_dir="./results/WebVid/pt/"
nproc_per_node=8 # determined by your resource



# script
# Pre-training on WebVid-2M and CC3M
python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
--config configs/pt/cc-webvid2m-pt-i2k.json --launcher pytorch \
--save_dir $save_dir --data_dir_0 $CC3M_root --data_dir_1 $WebVid_root \
--CLU true --CLU_learn_region true --CLU_learn_region_num 8 --CLU_region_att 'joint' \
--epochs 50 --schedule 30 40





if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi
