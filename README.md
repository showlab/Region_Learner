# Region_Learner
The Pytorch implementation for "Video-Text Pre-training with Learned Regions"
([arxiv](https://arxiv.org/pdf/2112.01194.pdf))

***We are still cleaning up the code further and preparing for pre-training weights.***

## Preparation
Overall, this code is built on PyTorch with DistributedDataParallel (DDP).
- Create conda env and install required packages via `sh setup_myEnv.sh`
- Create some important folders
	1. `mkdir data` (you can symlink huge datasets to this folder)
	2. `mkdir meta_data` (put meta data of each dataset here)
	3. `mkdir results`
- Download Pre-training data
	1. Download WebVid-2M (see https://github.com/m-bain/webvid)
	2. Download CC3M (see https://ai.google.com/research/ConceptualCaptions/download)

PS: Not all videos are available so you need to modify the metadata depending on your case. We also provide our metadata in [here](https://drive.google.com/drive/folders/1y9Byj2IFWSyeGiyJJwc2VPIESzakGHAh?usp=sharing).


## Pre-training 
- Run `sh pre-training.sh` (Commands with different settings are listed in this script.)

## Finetuning (on MSR-VTT)
- Download data (see https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt)
- Run `sh fine-tune.sh`.

## Pre-trained Weights
[WebVid2M + CC3M](https://drive.google.com/file/d/1ql5PDgaTqA9pQcBb1cYRGkH3IfbbUkSv/view?usp=sharing)

## Acknowledgements 
This code is based off [Frozen in Time](https://github.com/m-bain/frozen-in-time "Frozen in Time")





## Citation
```
@article{yan2021video,
  title={Video-Text Pre-training with Learned Regions},
  author={Yan, Rui and Shou, Mike Zheng and Ge, Yixiao and Wang, Alex Jinpeng and Lin, Xudong and Cai, Guanyu and Tang, Jinhui},
  journal={arXiv preprint arXiv:2112.01194},
  year={2021}
}
```