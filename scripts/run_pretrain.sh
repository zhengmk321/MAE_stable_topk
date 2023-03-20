#!/bin/bash
module load python3/3.9.7

which python3

hostfile="./hostfile"
if [ ! -d "/tmp/imagenet-1k" ] ; then
    echo "Decompressing imagenet-1k to /tmp/imagenet-1k now..."
    mpiexec.hydra -f $hostfile  -np 4 -ppn 1 mkdir /tmp/imagenet-1k
    mpiexec.hydra -f $hostfile  -np 4 -ppn 1 tar xf /scratch/00946/zzhang/data/imagenet/imagenet-1k.tar -C /tmp/imagenet-1k
fi

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
# echo $MASTER_ADDR

# Distributed training parameters
master_address=$( hostname -i )

# master_address="129.114.44.104"
master_port="1234"
# node_rank="1"

ngpus=3 # number of GPUs per node
nnodes=4 # number of total nodes
num_workers=$(($ngpus * $nnodes)) # total number of gpus on all nodes

# ViT model parameters
model="mae_vit_large_patch16"
task_name="pretrain_vit_large_patch16"
batch_size="256"
epoch="200"
start_epoch="0"
blr="1.5e-4"
weight_decay="0.05"
mask_ratio="0.75"
accum_iter="1"
epochs="100"

# oktopk parameters
compressor="topkA_stable"
density="0.01"
stable_topk_interval="500"

# Files/directories 
data_path="/tmp/imagenet-1k/ILSVRC2012_img_"
# data_path="/work/09308/zhengmk/imagenet-small/ILSVRC2012_img_"
output_dir="../results/"
resume=""

# CMD="python3 -m torch.distributed.launch "
CMD="mpiexec.hydra -f $hostfile  -np 12 -ppn 3 python3 -m mpi4py "
CMD+="/work/09308/zhengmk/ViT/mae_stable_topk/main_pretrain.py "
CMD+="--master_addr $master_address "
CMD+="--master_port $master_port "
CMD+="--model $model "
CMD+="--batch_size $batch_size "
CMD+="--epochs $epochs "
CMD+="--do_train "
CMD+="--density $density "
CMD+="--compressor $compressor "
CMD+="--accum_iter $accum_iter "
CMD+="--dataparallel "
CMD+="--fp16 "
CMD+="--norm_pix_loss "
CMD+="--weight_decay $weight_decay "
CMD+="--data_path $data_path "
CMD+="--output_dir $output_dir "
CMD+="--blr $blr "
CMD+="--stable_topk_interval $stable_topk_interval "

LOGFILE=$output_dir$task_name.log

echo $CMD
if [ $resume="" ] ; then
    echo "Starting a new pretrain"
    echo "$LOGFILE will be cleaned"
    $CMD |& tee $LOGFILE 
else
    echo "Continue pretraining from epoch $start_epoch"
    $CMD >> $LOGFILE 
fi
