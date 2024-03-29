# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import contextlib
from contextlib import contextmanager
import itertools
import time
import torch
#import torch.distributed as dist

import communication
import runtime_utilities
import os

from torch.cuda.amp import autocast, GradScaler

BERT = "bert"
GPT2 = "gpt2"
IMAGE_CLASSIFICATION = "image_classification"
SPEECH_TO_TEXT = "speech_to_text"
TRANSLATION = "translation"
ViT = "ViT"

class InputSourceBase():
    def __init__(self):
        pass

    def get_inputs(self):
        raise NotImplementedError

class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False


class StageRuntime:
    def __init__(self, model, fp16, loss_scale,
                 training_tensor_shapes, eval_tensor_shapes,
                 training_tensor_dtypes, inputs_module_destinations,
                 target_tensor_names, configuration_maps, master_addr,
                 rank, local_rank, num_ranks_in_server, verbose_freq,
                 model_type,
                 use_apex=False,
                 reverse=False):
        # Metadata needed for forward and backward pass within this stage.
        self.tensors = []
        self.gradients = {}
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names
        self.use_apex = use_apex
        self.reverse = reverse

        self.initialize(model, inputs_module_destinations, configuration_maps,
                        master_addr, rank, local_rank, num_ranks_in_server)

        self.verbose_freq = verbose_freq
        self.forward_only = False

        self.forward_stats = runtime_utilities.RuntimeStats(forward=True)
        self.backward_stats = runtime_utilities.RuntimeStats(forward=False)

        self.input_source = None

        self.epoch = 0
        self.counter = 0

    def initialize(self, model, inputs_module_destinations,
                   configuration_maps, master_addr, rank,
                   local_rank, num_ranks_in_server):
        self.send_ranks = {}
        self.receive_ranks = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags = {}
        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0
        self.criterion_input_name = str(model[-1][1][0])

        # if self.reverse:
        #     tensor_tag = 16384
        # else:
        #     tensor_tag = 1

        # for (_, input_tensors, output_tensors) in model:
        #     for input_tensor in input_tensors:
        #         if input_tensor not in self.tensor_tags:
        #             self.tensor_tags[input_tensor] = tensor_tag
        #             tensor_tag += 1
        #     for output_tensor in output_tensors:
        #         if output_tensor not in self.tensor_tags:
        #             self.tensor_tags[output_tensor] = tensor_tag
        #             tensor_tag += 1
        # for target_tensor_name in sorted(self.target_tensor_names):
        #     self.tensor_tags[target_tensor_name] = tensor_tag
        #     tensor_tag += 1
        # self.tensor_tags["ack"] = tensor_tag
        # tensor_tag += 1

        self.module_to_stage_map = configuration_maps['module_to_stage_map']
        self.stage_to_rank_map = configuration_maps['stage_to_rank_map']
        self.stage_to_depth_map = configuration_maps['stage_to_depth_map']

        if self.module_to_stage_map is None:
            #assert self.rank is None
            print("Data parallelism is used!")
            self.modules_with_dependencies = ModulesWithDependencies(
                [(stage_module_fn(), inputs, outputs)
                 for (stage_module_fn, inputs, outputs) in model])
            self.is_criterion = True
            self.rank_in_stage = self.rank

            rank_to_stage_map = {}
            for stage in self.stage_to_rank_map:
                for rank in self.stage_to_rank_map[stage]:
                    rank_to_stage_map[rank] = stage

            assert 0 <= self.rank < len(rank_to_stage_map)
            self.num_ranks = len(rank_to_stage_map)
            self.num_ranks_in_first_stage = len(rank_to_stage_map)
            self.num_ranks_in_previous_stage = 0
            self.num_ranks_in_next_stage = 0
            self.num_stages = 1
            self.stage = 0
            self.num_ranks_in_stage = len(rank_to_stage_map)
            self.num_warmup_minibatches = 0
            self.comm_handler = None
            #os.environ['MASTER_ADDR'] = master_addr
            #os.environ['MASTER_PORT'] = "1234"
            #dist.init_process_group("gloo", rank=self.rank, world_size=self.num_ranks)
            #assert dist.get_world_size() == self.num_ranks
        else:
            pass
            #assert len(self.module_to_stage_map) == len(model)
            #assert self.rank is not None

            #stage_to_module_map = collections.defaultdict(list)
            #for module in range(len(self.module_to_stage_map)):
            #    stage_to_module_map[self.module_to_stage_map[module]].append(module)

            #rank_to_stage_map = {}
            #for stage in self.stage_to_rank_map:
            #    for rank in self.stage_to_rank_map[stage]:
            #        rank_to_stage_map[rank] = stage

            ## Now, use this mapping to determine the modules contained in
            ## each stage.
            #assert 0 <= self.rank < len(rank_to_stage_map)
            #self.num_ranks = len(rank_to_stage_map)
            #self.num_stages = len(stage_to_module_map)
            #self.stage = rank_to_stage_map[self.rank]
            #self.rank_in_stage = self.stage_to_rank_map[self.stage].index(self.rank)
            #self.num_ranks_in_stage = len(self.stage_to_rank_map[self.stage])
            #self.num_ranks_in_first_stage = len(self.stage_to_rank_map[0])
            #self.num_ranks_in_previous_stage = 0
            #self.ranks_in_previous_stage = []
            #if self.stage > 0:
            #    self.num_ranks_in_previous_stage = len(
            #        self.stage_to_rank_map[self.stage - 1])
            #    self.ranks_in_previous_stage = self.stage_to_rank_map[self.stage - 1]
            #self.num_ranks_in_next_stage = 0
            #self.ranks_in_next_stage = []
            #if self.stage < self.num_stages - 1:
            #    self.num_ranks_in_next_stage = len(
            #        self.stage_to_rank_map[self.stage + 1])
            #    self.ranks_in_next_stage = self.stage_to_rank_map[self.stage + 1]
            #modules = stage_to_module_map[self.stage]
            #self.modules_with_dependencies = ModulesWithDependencies(
            #    [(model[module][0](), model[module][1], model[module][2]) for module in modules])
            #self.is_criterion = self.stage == (self.num_stages - 1)
            #if self.stage_to_depth_map is not None:
            #    self.num_warmup_minibatches = self.stage_to_depth_map[
            #        str(self.stage)]
            #else:
            #    self.num_warmup_minibatches = self.num_ranks - 1
            #    for i in range(self.stage):
            #        self.num_warmup_minibatches -= len(
            #            self.stage_to_rank_map[i])
            #    self.num_warmup_minibatches = self.num_warmup_minibatches // \
            #        self.num_ranks_in_stage

            ## To determine where tensors should be sent and received, first
            ## determine the "producing" and "consuming" module IDs of each
            ## tensor. We then use the corresponding machine ranks to send
            ## and receive tensors.
            #master_port = 1234
            #self.comm_handler = communication.CommunicationHandler(
            #    master_addr=master_addr,
            #    master_port=master_port,
            #    rank=self.rank,
            #    local_rank=self.local_rank,
            #    num_ranks_in_server=num_ranks_in_server,
            #    world_size=self.num_ranks,
            #    fp16=self.fp16,
            #    num_stages=self.num_stages,
            #    reverse=self.reverse)

            #for i in range(len(model)):
            #    for j in range(i+1, len(model)):
            #        for tensor_name in model[i][2]:
            #            if tensor_name in model[j][1]:
            #                if self.module_to_stage_map[i] == \
            #                    self.module_to_stage_map[j]:
            #                    continue
            #                # For now, assume that each stage is served by only
            #                # a single machine.
            #                index = self.stage_to_rank_map[self.stage].index(self.rank)
            #                if self.module_to_stage_map[j] == self.stage:
            #                    if len(self.stage_to_rank_map[self.stage]) == \
            #                        len(self.stage_to_rank_map[self.module_to_stage_map[i]]):
            #                        self.receive_ranks[tensor_name] = \
            #                            [self.stage_to_rank_map[self.module_to_stage_map[i]][index]]
            #                    else:
            #                        self.receive_ranks[tensor_name] = \
            #                            self.stage_to_rank_map[self.module_to_stage_map[i]]
            #                if self.module_to_stage_map[i] == self.stage:
            #                    if len(self.stage_to_rank_map[self.stage]) == \
            #                        len(self.stage_to_rank_map[self.module_to_stage_map[j]]):
            #                        self.send_ranks[tensor_name] = \
            #                            [self.stage_to_rank_map[self.module_to_stage_map[j]][index]]
            #                    else:
            #                        self.send_ranks[tensor_name] = \
            #                            self.stage_to_rank_map[self.module_to_stage_map[j]]

            #for model_inputs in inputs_module_destinations.keys():
            #    destination_stage = self.module_to_stage_map[
            #        inputs_module_destinations[model_inputs]]
            #    index = self.stage_to_rank_map[self.stage].index(self.rank)
            #    if destination_stage > self.stage:
            #        if len(self.stage_to_rank_map[self.stage]) == len(self.ranks_in_next_stage):
            #            self.send_ranks[model_inputs] = \
            #                [self.ranks_in_next_stage[index]]
            #        else:
            #            self.send_ranks[model_inputs] = \
            #                self.ranks_in_next_stage

            #    if 0 < self.stage <= destination_stage:
            #        if len(self.stage_to_rank_map[self.stage]) == len(self.ranks_in_previous_stage):
            #            self.receive_ranks[model_inputs] = \
            #                [self.ranks_in_previous_stage[index]]
            #        else:
            #            self.receive_ranks[model_inputs] = \
            #                self.ranks_in_previous_stage

            #    if destination_stage > 0:
            #        if model_inputs not in self.tensor_tags:
            #            self.tensor_tags[model_inputs] = tensor_tag
            #            tensor_tag += 1

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def initialize_distributed_backend(self):
        # Initialize all groups in the same order on every worker.
        #modules = self.modules_with_dependencies.modules()
        #if self.stage_to_rank_map is not None:
        #    groups = []
        #    for stage in range(self.num_stages):
        #        ranks = self.stage_to_rank_map[stage]
        #        if len(ranks) > 1:
        #            #groups.append(dist.new_group(ranks=ranks, backend='nccl'))
        #            groups.append(dist.new_group(ranks=ranks))
        #        else:
        #            groups.append(None)
        #    group = groups[self.stage]
        #else:
        #    group = None

        # self.modules_with_dependencies contains a list of PyTorch
        # modules, along with a list of user-defined input and output
        # tensor names. We use our module_executor.ModuleExecutor
        # class to wrap these dependencies, and use run_forward and
        # run_backward methods downstream.

        #num_parameters = 0
        #for i in range(len(modules)):
        #    if group is not None:
        #        if ((i < (len(modules)-1) and self.is_criterion)
        #            or not self.is_criterion):
        #            num_parameters += \
        #                sum(x.size()[0] * x.size()[1]
        #                    if len(x.size()) > 1 else x.size()[0]
        #                    for x in modules[i].parameters() if x.size())
        #            if self.num_stages == 1 and self.use_apex:
        #                import apex
        #                modules[i] = apex.parallel.DistributedDataParallel(
        #                    modules[i])
        #            else:
        #                modules[i] = torch.nn.parallel.DistributedDataParallel(
        #                    modules[i],
        #                    process_group=group,
        #                    device_ids=[self.local_rank],
        #                    output_device=self.local_rank)
        #if self.num_ranks_in_stage > 1:
        #    module_size = 4. * num_parameters
        #    print("Replicating stage: ranks=%d, module_size=%.3f" % (
        #        self.num_ranks_in_stage, module_size))

        self.master_parameters = list(self.parameters())

        #if self.comm_handler is not None:
        #    self.comm_handler.initialize(
        #        self.receive_ranks,
        #        self.send_ranks,
        #        self.tensor_tags,
        #        self.target_tensor_names,
        #        self.training_tensor_dtypes,
        #        self.rank_in_stage,
        #        self.num_ranks_in_stage,
        #        self.ranks_in_previous_stage,
        #        self.ranks_in_next_stage)

    @property
    def target(self):
        return self.tensors[-1]["target"]

    def is_first_stage(self):
        return self.stage is None or (self.stage == 0)

    def is_last_stage(self):
        return self.stage is None or (self.stage == (self.num_stages-1))

    def modules(self):
        return self.modules_with_dependencies.modules()

    def context_handlers(self):
        context_handlers = []
        modules = self.modules_with_dependencies.modules()
        for i, module in enumerate(modules):
            context_handler = self.dummy_handler()
            if (i < (len(modules)-1) and self.is_criterion) or not self.is_criterion:
                if self.num_ranks_in_stage > 1:
                    #context_handler = module.no_sync()
                    context_handler = self.dummy_handler()
            context_handlers.append(context_handler)
        return context_handlers

    def parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            parameter_iterators.append(module.parameters())
        return itertools.chain(*parameter_iterators)

    def named_parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            parameter_iterators.append(module.named_parameters())
        return itertools.chain(*parameter_iterators)

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            state_dict["module%d" % i] = module.state_dict()
        if self.fp16:
            state_dict["master_parameters"] = self.master_parameters
        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            module.load_state_dict(state_dict["module%d" % i])
        if self.fp16:
            saved_master_parameters = state_dict["master_parameters"]
            for master_parameter, saved_master_parameter in zip(
                self.master_parameters, saved_master_parameters):
                master_parameter.data.copy_(saved_master_parameter.data)

    def cuda(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=False)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=True)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def set_input_source(self, input_source):
        if not isinstance(input_source, InputSourceBase):
           raise Exception('input_source needs to be derived from runtime.InputSourceBase') 
        self.input_source = input_source

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    def receive_tensors_forward(self):
        if self.forward_only and len(self.tensors) > 0:
            self.tensors.pop(0)
        
        if self.is_first_stage():
            if self.input_source is None:
                raise Exception('input source is None in first stage')

            self.tensors.append(self.input_source.get_inputs())
            #print("original criterion inputs target_lm: ", self.tensors[-1]["target_lm"].view(-1), ", target_sentence: ", self.tensors[-1]["target_sentence"].view(-1))
        else:
            # Receive all required tensors from upstream machines.
            self.tensors.append({})
            for input_name in self.receive_ranks:
                if input_name == "ack":
                    continue

                self.tensors[-1][input_name] = \
                    self.comm_handler.recv(
                        input_name,
                        forward_minibatch_id=self.forward_minibatch_id,
                        backward_minibatch_id=self.backward_minibatch_id,
                        backward=False)

                self.forward_stats.stats['receive_tensors_size'] += \
                    (self.tensors[-1][input_name].element_size() *
                     self.tensors[-1][input_name].nelement())

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(
                sending=False)

    def send_tensors_forward(self):
        # Send all required tensors downstream.
        for output_name in self.send_ranks:
            if output_name == "ack":
                continue

            self.comm_handler.send(
                output_name,
                self.tensors[-1][output_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=False)

            self.forward_stats.stats['send_tensors_size'] += \
                (self.tensors[-1][output_name].element_size() *
                 self.tensors[-1][output_name].nelement())

    def receive_tensors_backward(self):
        # Receive all required gradients from downstream
        # machines.
        for output_name in self.send_ranks:
             if output_name in self.target_tensor_names:
                continue

             self.gradients[output_name] = \
                self.comm_handler.recv(
                    output_name,
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=True)

             self.backward_stats.stats['receive_tensors_size'] += \
                 (self.gradients[output_name].element_size() *
                  self.gradients[output_name].nelement())

    def send_tensors_backward(self):
        # Send all required gradients upstream.
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names:
                continue

            self.comm_handler.send(
                input_name,
                self.gradients[input_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            self.backward_stats.stats['send_tensors_size'] += \
                (self.gradients[input_name].element_size() *
                 self.gradients[input_name].nelement())

        if self.num_ranks_in_previous_stage > 0:
            # Used to track where to send tensors in the
            # backward pass.
            self.comm_handler.increment_messaging_index(
                sending=True)

    def run_forward(self, recompute_step=False):
        """Run forward pass.
        """
        # Receive tensors from previous worker.
        self.receive_tensors_forward()
        tensors = self.tensors[-1]
        if recompute_step:
            tensors = {}
            for (key, value) in self.tensors[-1].items():
                tensors[key] = value

        # Run forward pass.
        self._run_forward(tensors, recompute_step=recompute_step)
        if recompute_step:
            for output_name in self.send_ranks:
                if output_name == "ack":
                    continue
                self.tensors[-1][output_name] = tensors[output_name]

        # Send tensors forward.
        self.send_tensors_forward()
        if self.verbose_freq > 0 and self.forward_minibatch_id % self.verbose_freq == 0:
            self.forward_stats.print_stats()
        self.forward_stats.reset_stats()
        self.forward_minibatch_id += 1

    @contextmanager
    def dummy_handler(self):
        try:
            yield
        finally:
            pass

    def _run_forward(self, tensors, recompute_step=False):
        # Perform forward pass through model (self.modules_with_dependencies already
        # has modules in topological order).
        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        no_grad_context_handler = self.dummy_handler if not recompute_step else torch.no_grad
        with no_grad_context_handler():
            for i, (module, input_names, output_names) in \
                    enumerate(zip(modules, all_input_names, all_output_names)):
                if i == (len(modules) - 1) and self.is_criterion:
                    # If layer is criterion (loss).
                    if self.model_type == BERT:
                        #print("criterion: ", input_names[0], tensors[input_names[0]], "shapes: ", tensors[input_names[0]][0].view(-1, self.vocab_size).shape, ",", tensors["target_lm"].view(-1).shape, ",", tensors[input_names[0]][1].view(-1, 2).shape, ",", tensors["target_sentence"].view(-1).shape)
                        #print("criterion inputs target_lm: ", tensors["target_lm"].view(-1), ", target_sentence: ", tensors["target_sentence"].view(-1))
                        masked_lm_loss = module(
                            tensors[input_names[0]][0].view(
                                -1, self.vocab_size),
                            tensors["target_lm"].view(-1))
                        next_sentence_loss = module(
                            tensors[input_names[0]][1].view(-1, 2),
                            tensors["target_sentence"].view(-1))
                        loss = masked_lm_loss + next_sentence_loss
                        module_outputs = [loss]
                    elif self.model_type == GPT2:
                        output = tensors[input_names[0]][..., :-1, :].contiguous()
                        output = output.view(-1, output.size(-1))
                        shift_labels = tensors["labels"][..., 1:].contiguous()
                        loss = module(output, shift_labels.view(-1))
                        module_outputs = [loss]
                    elif self.model_type == SPEECH_TO_TEXT:
                        output = tensors["output"].transpose(0, 1).float()
                        output_sizes = tensors["output_sizes"].cpu()
                        target = tensors["target"].cpu()
                        target_sizes = tensors["target_length"].cpu()
                        input0_size = tensors["input0_size"].cpu()
                        module_outputs = [module(output, target, output_sizes,
                                                    target_sizes) / input0_size[0]]
                    elif self.model_type == ViT:
                        with autocast():
                            loss, _, _ = model(samples,mask_ratio=mask_ratio)
                    else:
                        module_outputs = [module(tensors[input_name],
                                                 tensors["target"])
                                          for input_name in input_names]
                        module_outputs = [sum(module_outputs)]
                else:
                    # If layer is non-criterion.
                    module_outputs = module(*[tensors[input_name]
                                              for input_name in input_names])
                    if not isinstance(module_outputs, tuple):
                        module_outputs = (module_outputs,)
                    module_outputs = list(module_outputs)

                if len(output_names) == 1 and len(module_outputs) > 1:
                    tensors[output_names[0]] = tuple(module_outputs)
                else:
                    for (output_name, module_output) in zip(output_names, module_outputs):
                        tensors[output_name] = module_output

            self.output = tensors[input_names[0]]
            if self.is_criterion and self.model_type == TRANSLATION:
                loss_per_batch = tensors[output_names[0]] * tensors[
                    self.criterion_input_name].size(1)
                loss_per_token = loss_per_batch / tensors[
                    "target_length"][0].item()
                self.loss = loss_per_token
            elif self.is_criterion:
                self.loss = tensors[output_names[0]]
            else:
                self.loss = 1

    def run_backward(self, recompute_step=False):
        # Receive input gradients needed for backward pass.
        self.receive_tensors_backward()
        # Backward pass through modules in reverse order.
        inputs = {}
        outputs = {}
        input_gradients = {}
        output_gradients = {}

        # Get input and output names spanning all modules in this stage.
        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)

        tensors = self.tensors.pop(0)
        if recompute_step:
            self._run_forward(tensors, recompute_step=False)

        # Set inputs, outputs, and output_gradients.
        # Only set outputs/output_gradients for tensors that are not inputs of
        # other modules in this stage.
        # Similarly, only set inputs for tensors that are not outputs of other
        # modules in this stage.
        for (module, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set:
                    inputs[input_name] = tensors[input_name]

        # Hook to record input gradients.
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        # Perform backward pass.
        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))

        # Input tensors don't need gradients.
        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = input_gradients[input_name]

        # Send output gradients.
        self.send_tensors_backward()
        if self.verbose_freq > 0 and self.backward_minibatch_id % self.verbose_freq == 0:
            self.backward_stats.print_stats()
        self.backward_stats.reset_stats()
        self.backward_minibatch_id += 1

    def _print_training_progress(self, step, n, start_time, epoch_start_time,
                                 loss, cumulative_loss, rank=-1):
        #if self.is_last_stage():
        cumu_loss = 0
        if self.is_last_stage() and (step+1) % 128 == 0 and len(cumulative_loss) != 0:  
        #if self.is_last_stage() and len(cumulative_loss) != 0:  
            cumu_loss = sum(cumulative_loss) / len(cumulative_loss)
            print("Step [%d/%d], Rank = %d, Time/iteration: %.3f seconds (%.3f seconds), Loss: %.3f (%.3f), Memory: %.3f GB (%.3f GB)" % (
                step, n, self.rank,
                (time.time() - start_time) / self.update_interval,
                (time.time() - epoch_start_time) / step,
                loss, cumu_loss,
                float(torch.cuda.max_memory_allocated()) / 10**9,
                float(torch.cuda.max_memory_reserved()) / 10**9))
                #float(torch.cuda.memory_allocated()) / 10**9,
                #float(torch.cuda.memory_cached()) / 10**9))

    def run_training_loop_with_1f1b_flushes(self, n, optimizer, recompute_step):
        cumulative_loss = []
        step = 0
        loss = None
        self.epoch += 1

        num_warmup_minibatches = self.num_warmup_minibatches
        self.train(n)
        epoch_start_time = time.time()
        start_time = time.time()
        #one iteration warmup
        with contextlib.ExitStack() as stack:
            for context_handler in self.context_handlers():
                stack.enter_context(context_handler)
            for wmf_step in range(num_warmup_minibatches):
                self.run_forward(recompute_step=recompute_step)

        for body_step in range(self.update_interval - num_warmup_minibatches - 1):

            with contextlib.ExitStack() as stack:
                for context_handler in self.context_handlers():
                    stack.enter_context(context_handler)

                self.run_forward(recompute_step=recompute_step)
                if self.is_last_stage():
                    loss = self.loss.item()
                    cumulative_loss.append(loss)

                self.run_backward(recompute_step=recompute_step)

        self.run_forward(recompute_step=recompute_step)
        if self.is_last_stage():
            loss = self.loss.item()
            cumulative_loss.append(loss)

        with contextlib.ExitStack() as stack:
            for context_handler in self.context_handlers():
                stack.enter_context(context_handler)
            for tailb_step in range(num_warmup_minibatches):
                self.run_backward(recompute_step=recompute_step)

        self.run_backward(recompute_step=recompute_step)

        optimizer.step()
        optimizer.zero_grad()

        if self.rank == self.num_ranks - 1:      
            self._print_training_progress(self.update_interval-1, n, start_time, epoch_start_time,
                                          loss, cumulative_loss)
        print("after warmup")
        #dist.barrier()
        epoch_start_time = time.time()
        for base_step in range(self.update_interval, n, self.update_interval):
            start_time = time.time()
            if base_step % 384 == 0:
                cumulative_loss = []

            with contextlib.ExitStack() as stack:
                for context_handler in self.context_handlers():
                    stack.enter_context(context_handler)
                for wmf_step in range(num_warmup_minibatches):
                    self.run_forward(recompute_step=recompute_step)

            for body_step in range(self.update_interval - num_warmup_minibatches - 1):

                with contextlib.ExitStack() as stack:
                    for context_handler in self.context_handlers():
                        stack.enter_context(context_handler)

                    self.run_forward(recompute_step=recompute_step)
                    if self.is_last_stage():
                        loss = self.loss.item()
                        cumulative_loss.append(loss)

                    self.run_backward(recompute_step=recompute_step)

            self.run_forward(recompute_step=recompute_step)
            if self.is_last_stage():
                loss = self.loss.item()
                cumulative_loss.append(loss)

            with contextlib.ExitStack() as stack:
                for context_handler in self.context_handlers():
                    stack.enter_context(context_handler)
                for tailb_step in range(num_warmup_minibatches):
                    self.run_backward(recompute_step=recompute_step)


            self.run_backward(recompute_step=recompute_step)

            optimizer.step()
            optimizer.zero_grad()


            if self.rank == self.num_ranks - 1:      
                self._print_training_progress(base_step-1, n, start_time, epoch_start_time,
                                              loss, cumulative_loss)

        #dist.barrier()
        print("Rank = ", self.rank, "Epoch = ", self.epoch, " update steps: ", (n/self.update_interval-1), ", Finish one epoch by 1f1b flush. Average time per update step: ", (time.time()-epoch_start_time)/(n/self.update_interval-1))


    def run_training_loop_with_flushes(self, n, optimizer, recompute_step):
        # NOTE: This does not work with replicated stages since Python's `no_sync()`
        # API is not intended for a computation pattern of a sequence of forward
        # passes followed by a sequence of backward passes. If run with replicated
        # stages, weight synchronization will happen every backward pass, leading
        # to poor performance.
        cumulative_loss = []
        loss = None
        self.epoch += 1
        self.train(n)
        epoch_start_time = time.time()
        start_time = time.time()

        for step in range(self.update_interval):
            self.run_forward(recompute_step=recompute_step)
            if self.is_last_stage():
                loss = self.loss.item()
                cumulative_loss.append(loss)

        for step in range(self.update_interval):
            self.run_backward(recompute_step=recompute_step)

        optimizer.step()
        optimizer.zero_grad()

        if self.rank == self.num_ranks - 1:      
            self._print_training_progress(self.update_interval, n, start_time, epoch_start_time,
                                          loss, cumulative_loss)
        #self._print_training_progress(step, n, start_time, epoch_start_time,
        #                              loss, cumulative_loss)

        print("after warmup")
        #dist.barrier()
        epoch_start_time = time.time()
        for base_step in range(self.update_interval, n, self.update_interval):
            start_time = time.time()
            if base_step % 384 == 0:
                cumulative_loss = []

            for step in range(self.update_interval):
                self.run_forward(recompute_step=recompute_step)
                if self.is_last_stage():
                    loss = self.loss.item()
                    cumulative_loss.append(loss)

            for step in range(self.update_interval):
                self.run_backward(recompute_step=recompute_step)

            optimizer.step()
            optimizer.zero_grad()

            if self.rank == self.num_ranks - 1:      
                self._print_training_progress(base_step, n, start_time, epoch_start_time,
                                              loss, cumulative_loss)

        #dist.barrier()
        #if self.counter > 0:
        print("Rank = ", self.rank, "Epoch = ", self.epoch, " update steps: ", (n/self.update_interval-1), "update interval: ", self.update_interval, ", Finish one epoch by DP. Average time per update step: ", (time.time()-epoch_start_time)/(n/self.update_interval-1))
        self.counter += 1

    def run_training_loop(self, n, optimizer, recompute_step, no_input_pipelining):
        cumulative_loss = []
        step = 0
        loss = None
        if no_input_pipelining:
            num_warmup_minibatches = 0
        else:
            num_warmup_minibatches = self.num_warmup_minibatches
        self.train(n)

        start_time = time.time()
        epoch_start_time = time.time()
        with contextlib.ExitStack() as stack:
            for context_handler in self.context_handlers():
                stack.enter_context(context_handler)
            for step in range(num_warmup_minibatches):
                optimizer.load_forward_params()
                self.run_forward(recompute_step=recompute_step)

        for base_step in range(0, n - num_warmup_minibatches, self.update_interval):
            start_time = time.time()
            step = base_step
            with contextlib.ExitStack() as stack:
                for context_handler in self.context_handlers():
                    stack.enter_context(context_handler)
                num_steps_to_process = min(
                    self.update_interval,
                    n - num_warmup_minibatches - base_step)
                for step in range(base_step, base_step+num_steps_to_process-1):
                    if not no_input_pipelining:
                        optimizer.load_forward_params()
                    self.run_forward(recompute_step=recompute_step)

                    if self.is_last_stage():
                        loss = self.loss.item()
                        cumulative_loss.append(loss)

                    if not no_input_pipelining:
                        optimizer.load_backward_params()
                    self.run_backward(recompute_step=recompute_step)

            if not no_input_pipelining:
                optimizer.load_forward_params()
            self.run_forward(recompute_step=recompute_step)

            if self.is_last_stage():
                loss = self.loss.item()
                cumulative_loss.append(loss)

            if not no_input_pipelining:
                optimizer.load_backward_params()
            self.run_backward(recompute_step=recompute_step)

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            self._print_training_progress(step, n, start_time, epoch_start_time,
                                          loss, cumulative_loss)

        with contextlib.ExitStack() as stack:
            for context_handler in self.context_handlers():
                stack.enter_context(context_handler)
            for step in range(n - num_warmup_minibatches, n):
                optimizer.load_backward_params()
                self.run_backward(recompute_step=recompute_step)
            optimizer.step()
            optimizer.zero_grad()

        print("Time needed for %d iterations: %.2f seconds" % (
            n, time.time() - epoch_start_time))

    def num_tokens(self):
        return self.tensors[-1]["target_length"][0].item()

    def run_ack(self):
        # No need for ack if running on a single worker.
        #if self.rank is None:
        if self.comm_handler is None:
            return

        # Receive ack from next stage. Send ack to previous stage.
        if self.stage < (self.num_stages-1):
            self.comm_handler.recv(
                "ack",
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)
        if self.stage > 0:
            self.comm_handler.send(
                "ack",
                torch.zeros(self.tensor_shapes["ack"],
                            dtype=torch.int64).cuda(),
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(sending=True)

        self.backward_minibatch_id += 1

    def wait(self):
        if self.comm_handler is not None:
            self.comm_handler.wait()

    def num_iterations(self, loader_size):
        """ Determines number of iterations for this stage

        TODO: don't currently support uneven configurations.
        """
        if self.stage == 0 or self.stage is None:
            return loader_size

        num_iterations = loader_size * self.num_ranks_in_first_stage
        assert num_iterations % self.num_ranks_in_stage == 0
        num_iterations = num_iterations // self.num_ranks_in_stage

        return num_iterations

    def get_adjusted_learning_rate(self, base_lr):
        if self.stage == 0:
            return base_lr

        adjusted_lr = float(base_lr) * float(self.num_ranks_in_stage) \
                      / float(self.num_ranks_in_first_stage)

        return adjusted_lr
