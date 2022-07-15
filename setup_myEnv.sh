#  Our code is run with CUDA11 on 16 A100

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
