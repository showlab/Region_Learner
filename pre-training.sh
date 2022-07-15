
# install environments
sh setup_myEnv.sh



# goto workdir
# echo "load path ....."
cd your_path/Region_Learner





nproc_per_node=1 # determined by your resource
data_root="data"





# NOTE: Not all videos can be download. Maybe you need adjust the meta_data for each dataset class defined in 'data_loader'.
WebVid_root=$data_root"/WebVid"
CC3M_root=$data_root"/CC3M/"
save_dir="./results/pt/"


# new script
# Pre-training on WebVid-2M and CC3M
# NOTES: It will takes several minutes before the first epoch. Training speed depends on status of your IO system.
python -m torch.distributed.launch --nproc_per_node $nproc_per_node $@ train.py \
--config configs/pt/CC3M-WebVid2M.json --launcher pytorch \
--save_dir $save_dir --data_dir_0 $CC3M_root --data_dir_1 $WebVid_root \
--epochs 50 --schedule 30 40





if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi
