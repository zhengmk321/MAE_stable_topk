# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
from apex.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
from transformers.utils import is_main_process
import util.misc as misc
#import torch.distributed as distrib

multi_tensor_l2norm = amp_C.multi_tensor_l2norm
lamb_compute_update = amp_C.multi_tensor_lamb_stage1_cuda
lamb_apply_update = amp_C.multi_tensor_lamb_stage2_cuda
scale = amp_C.multi_tensor_scale

from compression import compressors
import allreducer as ar
import time
from datetime import datetime

from mpi4py import MPI
comm = MPI.COMM_WORLD
world_size = MPI.COMM_WORLD.size

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x - 1. )/ (warmup - 1.), 0.)
    
def warmup_poly(x, warmup=0.002, degree=0.5):
    if x < warmup:
        return x/warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}

# class ViTAdam(Optimizer):
#     """Implements ViT version of Adam algorithm with weight decay fix.
#     Params:
#         lr: learning rate
#         warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
#         t_total: total number of training steps for the learning
#             rate schedule, -1  means constant learning rate. Default: -1
#         schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
#         b1: Adams b1. Default: 0.9
#         b2: Adams b2. Default: 0.999
#         e: Adams epsilon. Default: 1e-6
#         weight_decay: Weight decay. Default: 0.01
#         max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
#     """
#     def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
#                  b1=0.9, b2=0.95, e=1e-5, weight_decay=0.05,
#                  max_grad_norm=1.0, flush_group=None, flush_group_size=None, stage_id=None, density=1.0, compressor='none', rank=-1):
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if schedule not in SCHEDULES:
#             raise ValueError("Invalid schedule parameter: {}".format(schedule))
#         if not 0.0 <= warmup < 1.0 and not warmup == -1:
#             raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
#         if not 0.0 <= b1 < 1.0:
#             raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
#         if not 0.0 <= b2 < 1.0:
#             raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
#         if not e >= 0.0:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
#         defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
#                         b1=b1, b2=b2, e=e, weight_decay=weight_decay,
#                         max_grad_norm=max_grad_norm)

#         self.flush_group = flush_group
#         self.flush_group_size = flush_group_size

#         self.stage_id = stage_id
#         self.rank = rank

#         self.grad_shapes = []
#         self.grad_sizes = []
#         self.counter = 0
#         if compressor == 'none':
#             is_sparse = False
#         else:
#             is_sparse = True

#         self.compressor = compressors[compressor]
#         self.allreducer = ar.AllReducer(compression=self.compressor, sparse=is_sparse, density=density)

#         super(ViTAdam, self).__init__(params, defaults)

#     def get_lr(self):
#         lr = []
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 if len(state) == 0:
#                     return [0]
#                 if group['t_total'] != -1:
#                     schedule_fct = SCHEDULES[group['schedule']]
#                     lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
#                 else:
#                     lr_scheduled = group['lr']
#                 lr.append(lr_scheduled)
#         return lr

#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         grads = []

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 if p.grad.data.is_sparse:
#                     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
#                 if self.counter == 0:
#                     self.grad_shapes.append(p.grad.data.shape)
#                     self.grad_sizes.append(torch.numel(p.grad.data))
#                 grads.append(p.grad.data.view(-1))
        
#         self.counter += 1
#         all_grads = torch.cat(grads)
#         #print("total gradients size of allreduce: ", torch.numel(all_grads), "number of gradients: ", len(self.grad_shapes), "gradients sizes: ", self.grad_sizes)
#         start_time = time.time()
#         all_grads_red = self.allreducer.run(all_grads)
    
#         print("counter: ", self.counter, "allreducer time: ", (time.time()-start_time))

#         split_all_grads = torch.split(all_grads_red, self.grad_sizes)

#         indx = 0
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 grad = split_all_grads[indx].view(self.grad_shapes[indx])
#                 assert grad.shape == p.grad.data.shape

#                 indx += 1

#                 state = self.state[p]
#                 # State initialization.
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values.
#                     state['next_m'] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values.
#                     state['next_v'] = torch.zeros_like(p.data)

#                 next_m, next_v = state['next_m'], state['next_v']
#                 beta1, beta2 = group['b1'], group['b2']

#                 ## Add grad clipping.
#                 # if group['max_grad_norm'] > 0:
#                 #     clip_grad_norm_(p, group['max_grad_norm'])
                
#                 # Decay the first and second moment running average coefficient.
#                 # In-place operations to update the averages at the same time.
#                 next_m.mul_(beta1).add_(1 - beta1, grad)
#                 next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                 update = next_m / (next_v.sqrt() + group['e'])

#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 if group['weight_decay'] > 0.0:
#                     update += group['weight_decay'] * p.data

#                 if group['t_total'] != -1:
#                     schedule_fct = SCHEDULES[group['schedule']]
#                     lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
#                 else:
#                     lr_scheduled = group['lr']

#                 # lr_scheduled = group['lr']

#                 update_with_lr = lr_scheduled * update
#                 p.data.add_(-update_with_lr)

#                 state['step'] += 1


#         return loss

from apex.multi_tensor_apply import multi_tensor_applier



import math
import torch
from torch import Tensor
from typing import List, Optional

__all__ = ['AdamW', 'adamw']

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        capturable (bool, optional): whether this instance is safe to capture in a CUDA graph.
            Passing True can impair ungraphed performance, so if you don't intend to
            graph capture this instance, leave it False (default: False)

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False, compressor='none', density=1.0,
                 stable_topk_interval=100, stable_topk_threshold=100, 
                 stable_topk_warmup_method='none', rank=-1,):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        
        self.rank = rank
        if compressor == 'none':
            is_sparse = False
        else:
            is_sparse = True
        self.counter = 0
        self.grad_shapes = []
        self.grad_sizes = []
        self.compressor = compressors[compressor]
        self.allreducer = ar.AllReducer(compression=self.compressor, 
                                        sparse=is_sparse, 
                                        density=density, 
                                        stable_topk_interval=stable_topk_interval, 
                                        stable_topk_threshold=stable_topk_threshold, 
                                        stable_topk_warmup_method=stable_topk_warmup_method)

        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grads_reshaped = []
        
        # Gather all the gradients and perform oktopk on them
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                if self.counter == 0:
                    self.grad_shapes.append(p.grad.data.shape)
                    self.grad_sizes.append(torch.numel(p.grad.data))
                grads_reshaped.append(p.grad.data.view(-1)) 

        self.counter += 1
        all_grads = torch.cat(grads_reshaped)

        start_time = time.time()
        assert all_grads is not None and len(all_grads) > 0
        all_grads_red = self.allreducer.run(all_grads)
        allreduce_time = time.time() - start_time

        avg_allreduce_time = comm.allreduce(allreduce_time, op=MPI.SUM)/world_size

        if self.rank == 0:
            print(f"Global step: [{self.counter-1}] Average allreducer time: {avg_allreduce_time:.3f}s")

        split_all_grads = torch.split(all_grads_red, self.grad_sizes)
        
        
        # Real optimization starts from here
        idx = 0
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                # grads.append(p.grad)
                grad = split_all_grads[idx].view(self.grad_shapes[idx])
                assert grad.shape == p.grad.data.shape
                grads.append(grad)
                

                idx += 1

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

            adamw(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'])

        return loss



def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[Tensor],
          # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
          # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
          foreach: bool = None,
          capturable: bool = False,
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    # if foreach and not torch.jit.is_scripting():
    #     func = _multi_tensor_adamw
    # else:
    #     func = _single_tensor_adamw

    func = _single_tensor_adamw

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable)


def _single_tensor_adamw(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = step_t.item()

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)


def _multi_tensor_adamw(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool):
    if len(params) == 0:
        return

    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avg_sqs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]

    # update steps
    torch._foreach_add_(state_steps, 1)

    # Perform stepweight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

    if capturable:
        # TODO: use foreach_pow if/when foreach_pow is added
        bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
        bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
        # foreach_sub doesn't allow a scalar as the first arg
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)

        # foreach_div doesn't allow a scalar as the first arg
        step_size = torch._foreach_div(bias_correction1, lr)
        torch._foreach_reciprocal_(step_size)
        torch._foreach_neg_(step_size)

        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
            torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

        torch._foreach_addcdiv_(params, exp_avgs, denom)
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]

        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

        torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)