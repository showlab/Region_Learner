# Region_Learner
The Pytorch implementation for "Video-Text Pre-training with Learned Regions"
([arxiv](https://arxiv.org/pdf/2112.01194.pdf))

***We are still cleaning up the code further and preparing for pre-training weights.***

## Preparation
Overall, this code is built on PyTorch with DistributedDataParallel (DDP).
- Create conda env and install required packages via `sh install_env.sh`
- Create some important folders
	1. `mkdir data` (you can symlink huge datasets to this folder)
	2. `mkdir results`

## Finetuning (on MSR-VTT)
- Download data (see https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt)
- Run `sh finetune.sh`

## Pre-training 
- Download WebVid-2M (see https://github.com/m-bain/webvid)
- Download CC-3M (see https://ai.google.com/research/ConceptualCaptions/download)
- Run `sh pre-training.sh`

## Pre-trained Weights
Coming soon.

## Acknowledgements 
This code is based off [Frozen in Time](https://github.com/m-bain/frozen-in-time "Frozen in Time")





