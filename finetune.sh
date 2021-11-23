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
MSRVTT_root="data/MSRVTT/"
MSRVTT_save_dir="./results/WebVid/ft/MSRVTT/"
CCWV_1f_pti2k_mw="./results/WebVid/pt/CC3M-WebVid2M-1f-pti2k/XXX/model_best.pth" # 'XXX' is the timestamp


nproc_per_node=4 # determined by your resource

# Pre-training on WebVid-2M and CC3M

python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train_dist_multi.py \
--config configs/ft/msrvtt_8f_i21k.json --launcher pytorch \
--save_dir $MSRVTT_save_dir --load_checkpoint $CCWV_1f_pti2k_mw \
--data_dir_0 $MSRVTT_root --learning_rate1 3e-5 --schedule 101



if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi
