# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import logging
import os
import random
import re
from io import open
import time

import apex.amp as amp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from transformers.optimization import AdamW, warmup_linear

from torch.utils.data import Dataset
import random

import sys
sys.path.append("..")
import runtime
from torch.optim.optimizer import required
from torch.cuda.amp import autocast, GradScaler
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from ptflops import get_model_complexity_info

import os.path as osp
import shlex
import signal
import subprocess
import threading
from typing import Any, Optional, Tuple

import ifcfg
#import torch.distributed as distrib

import timm.optim.optim_factory as optim_factory

import models_mae
import util.lr_sched as lr_sched
import util.misc as misc

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)

EXIT = threading.Event()
EXIT.clear()
REQUEUE = threading.Event()
REQUEUE.clear()

import warnings
warnings.simplefilter("ignore")

# # Default port to initialized the TCP store on
# DEFAULT_PORT = 12345
# # Default address of world rank 0
# DEFAULT_MASTER_ADDR = "127.0.0.1"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
INTERRUPTED_STATE_FILE = osp.join(
    os.environ["HOME"], ".interrupted_states", f"{SLURM_JOBID}.pth"
)
# os.environ['MASTER_PORT'] = "12345"

# Helper methods.

def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


def _requeue_handler(signal, frame):
    EXIT.set()
    REQUEUE.set()


def add_signal_handlers():
    signal.signal(signal.SIGINT, _clean_exit_handler)
    signal.signal(signal.SIGTERM, _clean_exit_handler)

    # SIGUSR2 can be sent to all processes to have them cleanup
    # and exit nicely.  This is nice to use with SLURM as scancel <job_id>
    # sets a 30 second timer for the job to exit, and it can take more than
    # 30 seconds for the job to cleanup and exit nicely.  When using NCCL,
    # forcing the job to exit without cleaning up can be bad.
    # scancel --signal SIGUSR2 <job_id> will set no such timer and will give
    # the job ample time to cleanup and exit.
    signal.signal(signal.SIGUSR2, _clean_exit_handler)

    signal.signal(signal.SIGUSR1, _requeue_handler)


def save_interrupted_state(state: Any, filename: str = None):
    r"""Saves the interrupted job state to the specified filename.
        This is useful when working with preemptable job partitions.

    This method will do nothing if SLURM is not currently being used and the filename is the default

    :param state: The state to save
    :param filename: The filename.  Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"
    """
    if SLURM_JOBID is None and filename is None:
        logger.warn("SLURM_JOBID is none, not saving interrupted state")
        return

    if filename is None:
        filename = INTERRUPTED_STATE_FILE

    torch.save(state, filename)


def load_interrupted_state(filename: str = None) -> Optional[Any]:
    r"""Loads the saved interrupted state

    :param filename: The filename of the saved state.
        Defaults to "${HOME}/.interrupted_states/${SLURM_JOBID}.pth"

    :return: The saved state if the file exists, else none
    """
    if SLURM_JOBID is None and filename is None:
        return None

    if filename is None:
        filename = INTERRUPTED_STATE_FILE

    if not osp.exists(filename):
        return None

    return torch.load(filename, map_location="cpu")


def requeue_job():
    r"""Requeues the job by calling `scontrol requeue ${SLURM_JOBID}`
    """
    if SLURM_JOBID is None:
        return

    if not REQUEUE.is_set():
        return

    #distrib.barrier()
    comm.Barrier()

    #if distrib.get_rank() == 0:
    if comm.rank == 0:
        logger.info(f"Requeueing job {SLURM_JOBID}")
        subprocess.check_call(shlex.split("scontrol requeue {SLURM_JOBID}"))


def get_ifname():
    return ifcfg.default_interface()["device"]

def init_distrib_slurm(args):
    r"""Initializes torch.distributed by parsing environment variables set
        by SLURM when `srun` is used or by parsing environment variables set
        by torch.distributed.launch

    :param backend: Which torch.distributed backend to use

    :returns: Tuple of the local_rank (aka which GPU to use for this process)
        and the TCPStore used for the rendezvous
    """
    assert (
        torch.distributed.is_available()
    ), "torch.distributed must be available"

    #if "GLOO_SOCKET_IFNAME" not in os.environ:
    #    os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    #if "NCCL_SOCKET_IFNAME" not in os.environ:
    #    os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    master_port = int(os.environ.get("MASTER_PORT", args.master_port))
    master_addr = os.environ.get("MASTER_ADDR", args.master_addr)
    # Check to see if we should parse from torch.distributed.launch
    if os.environ.get("LOCAL_RANK", None) is not None: # run with torchrun
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"]) # real rank, not node rank. ranging from [0, world_size]
        world_size = int(os.environ["WORLD_SIZE"])
        local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    # Else parse from SLURM is using SLURM
    else: # run with ibrun
        local_rank = int(os.environ["MPI_LOCALRANKID"])
        # world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["PMI_SIZE"])
        local_size = int(os.environ["MPI_LOCALNRANKS"])
        world_rank = comm.Get_rank() # consistent to world_rank in torchrun (i.e. real rank in world)
        
        # print(os.environ)
    return local_rank, local_size, world_rank, world_size, master_addr


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    ## Required parameters
    parser.add_argument("--partitioned_dataset",
                        action='store_true',
                        default=False,
                        help="Is the dataset partitioned into a number of files.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Batch size for training on one GPU (Effective batch size = batch_size * accum_iter * world_size).")
    parser.add_argument("--lr",
                        #default=3e-5,
                        default=2e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument("--density",
                        default=1.0,
                        type=float,
                        help="The density of the gradients.")
    parser.add_argument('--compressor', type=str, default='none')
    parser.add_argument("--epochs",
                        default=800,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")

    # PipeDream runtime parameters
    parser.add_argument('--model', '-m', required=True,default='mae_vit_large_patch16', type=str,
                        help='name of module that contains model and tensor_shapes definition')
    parser.add_argument('--master_addr', default=None, type=str,
                        help="IP address of master (machine with rank 0)")
    parser.add_argument('--master_port', default="1234", type=str,
                        help="Port number")
    
    parser.add_argument('--rank', default=None, type=int,
                        help="Rank of worker")
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='path to directory to save checkpoints')
    # GPipe-style execution.
    parser.add_argument('--dataparallel', action='store_true',
                        help='Use GPipe-style weight updates')

    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument('--accum_iter',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--use_apex',
                        action='store_true',
                        help="Use Apex for data-parallel communication among stages")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    # parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--model_dir', type=str, default='')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument("--stable_topk_interval",
                        default=100,
                        type=int,
                        help="How often to sample topk elements.")
    parser.add_argument("--stable_topk_threshold",
                        default=100,
                        type=int,
                        help="When to begin stable topk scheme.")
    parser.add_argument("--stable_topk_warmup_method",
                        default='none',
                        type=str,
                        help="Which topk method to use until stable_topk_threshold steps. Defaults to dense.")
    return parser

def main():
    
    args = get_args_parser()
    args = args.parse_args()

    local_rank, local_size, world_rank, world_size, master_addr = init_distrib_slurm(args)
    # print(f"local_rank: {local_rank}, local_size: {local_size}, world_rank {world_rank}, world_size: {world_size}, master_addr: {master_addr}")

    n_gpu = torch.cuda.device_count()
    assert local_size <= n_gpu
    args.local_rank = local_rank
    args.rank = world_rank
    args.master_addr = master_addr

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # print("device: {} local_size: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, local_size, bool(world_size != 1), args.fp16))


    if args.accum_iter < 1:
        raise ValueError("Invalid accum_iter parameter: {}, should be >= 1".format(
                            args.accum_iter))

    seed = args.seed + world_rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    if local_size > 0:
        torch.cuda.manual_seed_all(args.seed+world_rank)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    
    dataset_train = datasets.ImageFolder(args.data_path+'train', transform=transform_train)
    
    num_tasks, global_rank = world_size, world_rank
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    args.num_workers = world_size
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    num_train_optimization_steps=len(data_loader_train) * args.epochs
    
    
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if world_rank == 0: 
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

    # # Prepare optimizer.
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]

    optimizer_grouped_parameters = optim_factory.add_weight_decay(model, args.weight_decay)

    # optimizer = ViTAdam(optimizer_grouped_parameters,
    #                     lr=args.lr,
    #                     warmup=-1,
    #                     t_total=-1,
    #                     density=args.density,
    #                     compressor=args.compressor,
    #                     rank=args.rank)

    # optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95), adam_w_mode=True)

    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.lr, 
                      betas=(0.9, 0.95), 
                      density=args.density, 
                      compressor=args.compressor,
                      stable_topk_interval=args.stable_topk_interval,
                      stable_topk_threshold=args.stable_topk_threshold,
                      stable_topk_warmup_method=args.stable_topk_warmup_method,
                      rank = args.rank)

    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
    #                   lr=args.lr, 
    #                   betas=(0.9, 0.95), )

    loss_scaler = NativeScaler()

    if args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    # torch.cuda.reset_peak_memory_stats()
    # # print("before train")

    global_step = 0
    print_freq = 10
    save_ckpt_freq = 20
    if args.do_train:
        start_time = time.time()
        if args.local_rank != -1:
            comm.Barrier()
        for epoch in range(args.epochs):
            data_loader_train.sampler.set_epoch(epoch)
            model.train(True)
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
            header = 'Epoch: [{}]'.format(epoch)
            for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
                optimizer.zero_grad()
                samples = samples.to(device, non_blocking=True)
                # if world_rank == 0: print("Performs forward pass...")
                with autocast():
                    loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                
                loss_value = loss.item()
                
                loss /= args.accum_iter
                # if world_rank == 0: print("Performs backward pass...")
                
                loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % args.accum_iter == 0)

                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)
                
                if args.local_rank != -1:
                    comm.Barrier()
                
                metric_logger.update(loss=loss_value)

                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, 'time':time.time()-start_time,}

            if args.output_dir and (epoch % save_ckpt_freq == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            
            if args.output_dir and args.rank == 0:
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if args.rank != -1:
                comm.Barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    return

if __name__ == "__main__":
    main()
