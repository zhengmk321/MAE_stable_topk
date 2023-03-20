# MAE with stable_topk
This is the repo for [masked autoencoders (MAE)](https://github.com/facebookresearch/mae) with stable_topk. 
# How to run it
The main shell script is [./script/run_pretrain.sh](). Before you run it, please run the following commandas to generate a hostfile for mpiexec:
```
cd ./scripts
squeue -u $USER # Get the name of the nodes for current job (e.g. c301-[002-003])
scontrol show  hostname c301-[002-003] > hostfile
```
To change the compressor, please change ``compressor`` in [./script/run_pretrain.sh]()run_pretrain.sh.
# What to expect
The runtime information (e.g. learning rate, time, communication time, and loss for each step) will be collected into $model.log, and the ckpt file will be preserved every 20 epochs. To change the frequency of showing runtime information, please change ``print_freq`` in [main_pretrain.py](). To change the  frequency of saving ckpt files, please change ``save_ckpt_freq`` in [main_pretrain.py](). 
# Adapt sparsification to your own optimizer
To use sparse compressor for your own optimizer, simply gather all the gradients and use the allreducer to compress them before the optimizer starts to use them to perform updates on model weights. Check [./transformers/optimization.py]() for more details.