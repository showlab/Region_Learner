
# install environments
sh setup_myEnv.sh


# goto workdir
# echo "load path ....."
cd your_path/Region_Learner


nproc_per_node=4 # determined by your resource


# NOTE: Not all videos can be download. Maybe you need adjust the meta_data for each dataset class defined in 'data_loader'.

##################################### MSRVTT ###################################
# set your path
MSRVTT_root="data/MSRVTT/"
MSRVTT_save_dir="./results/ft/MSRVTT/"
CCWV_mw="your_path_to/model_best.pth" 

# fine-tuning on MSRVTT
python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train.py \
--config configs/ft/MSRVTT_8f.json --launcher pytorch \
--save_dir $MSRVTT_save_dir --load_checkpoint $CCWV_mw \
--data_dir_0 $MSRVTT_root --learning_rate1 3e-5 --schedule 101
#################################################################################



######################### TODO:Other benchmarks ##################################



if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi

