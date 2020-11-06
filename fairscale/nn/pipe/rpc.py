# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank

from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group

from . import Pipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors

PipeModel: Pipe
PipeResult: TensorOrTensors


SizeOrSizes = Union[torch.Size, List[torch.Size]]
DtypeOrDtypes = Union[torch.dtype, List[torch.dtype]]


def set_device_based_on_group(group: ProcessGroup) -> None:
    # torch.cuda.set_device(group.rank() % torch.cuda.device_count())
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())


def get_shapes(tensor: TensorOrTensors) -> SizeOrSizes:
    if isinstance(tensor, torch.Tensor):
        return tensor.shape
    else:
        return [t.shape for t in tensor]


def get_dtype(tensor: TensorOrTensors) -> DtypeOrDtypes:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    else:
        return [t.dtype for t in tensor]


def get_global_ranks_from_group(group: ProcessGroup) -> List[int]:
    return [_get_global_rank(group, r) for r in range(group.size())]


class PipeBackRedirect(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(ctx, inputs, dest, message, transport, futures, model):
        ctx.dest = dest
        ctx.message = message
        ctx.transport = transport
        ctx.futures = futures
        ctx.model = model
        return inputs

    @staticmethod
    # type: ignore
    def backward(ctx, *grad):
        ctx.message.tensors = tuple(grad)
        ctx.transport.send_message(ctx.message, sync=False, skip_header=True)
        ctx.model.back_helper([])

        torch.futures.wait_all(ctx.futures)
        return (None, None, None, None, None, None)


def callback_with_model(callback: Callable[[Any, Pipe], None], ctx: Any) -> None:
    try:
        group = get_pipeline_parallel_group()  # FIXME(tom) handle dynamic group
        set_device_based_on_group(group)

        with PipeModel.lock:
            callback(ctx, PipeModel)
    except Exception as e:
        print(f"Exception raised in rpc callback: {e}")
        raise e


class PipeRPCWrapper(nn.Module):
    """A wrapper for Pipe to control the entire pipeline from a single process.
    Typical usecase would have rank 0 construct `PipeRPCWrapper` and run the
    training loop as normal, and all other ranks would call
    `torch.distributed.rpc.shutdown()`

    To run code on each worker, e.g. to run the optimizer, use `foreach_worker`
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.group = cast(ProcessGroup, kwargs.get("group")) or get_pipeline_parallel_group()
        assert self.group.rank() == 0

        if True:
            assert (
                self.group == get_pipeline_parallel_group()
            ), "Can't pickle groups, so group must be `get_pipeline_parallel_group()`"
            kwargs["group"] = None
        else:
            kwargs["group"] = self.group

        kwargs["style"] = Pipe.AsyncSchedule
        kwargs["input_device"] = torch.device("cuda", torch.cuda.current_device())

        self.model = Pipe(*args, **kwargs)
        self.worker_map = kwargs["worker_map"]
        torch.futures.wait_all(self._foreach_worker(self._register_remote_model, args=(args, kwargs)))
        self.model.cuda()

    def _get_rpc_name(self, rank: int) -> str:
        return self.worker_map[_get_global_rank(self.group, rank)]

    def _foreach_worker(self, callback: Callable, args: Any = None) -> List[torch.futures.Future]:
        return [rpc.rpc_async(self._get_rpc_name(rank), callback, args=args) for rank in range(1, self.group.size())]

    def foreach_worker(
        self, callback: Callable[[Any, Pipe], None], ctx: Any = None, *, include_self: bool = False
    ) -> None:
        """Call `callback` on each worker with the `ctx` and model local to that
        worker. e.g.
        def register_optimizer(ctx, model):
            args, kwargs = ctx
            model.optimizer = torch.optim.SGD(model.parameters(), *args, **kwargs)

        pipe_model = PipeRPCWrapper( ... )

        pipe_model.foreach_worker(
            register_optimizer,
            ([], {"lr" : 0.01, "momentum" : 0.9})
        )
        """

        torch.futures.wait_all(self._foreach_worker(callback_with_model, args=(callback, ctx)))

        if include_self:
            with self.model.lock:
                callback(ctx, self.model)

    def forward(self, tensor: TensorOrTensors) -> TensorOrTensors:  # type: ignore
        shape = get_shapes(tensor)
        dtype = get_dtype(tensor)

        if isinstance(tensor, torch.Tensor):
            num_tensors = 1
        else:
            num_tensors = len(tensor)

        futures = self._foreach_worker(self._model_forward, args=(self.model.training, shape, dtype))

        if self.model.final_stage:
            return self.model(tensor)
        else:
            self.model(tensor)

            shape, dtype = futures.pop().wait()
            dest_rank = self.group.size() - 1
            dest = self._get_rpc_name(dest_rank)
            dest_global_rank = _get_global_rank(self.group, dest_rank)
            src_global_rank = torch.distributed.get_rank()
            queue = EVENT_LOOP_QUEUE

            activations = PipeMessage(dest_global_rank, src_global_rank, queue_name=queue, tensor_count=num_tensors)
            grads = PipeMessage(src_global_rank, dest_global_rank, queue_name=queue, tensor_count=num_tensors)

            back_fut = rpc.rpc_async(
                dest, self._send_result_and_do_backwards, args=(self.model.training, activations, grads)
            )
            futures.append(back_fut)

            result = self._recv_result(shape, dtype, activations)
            if isinstance(result, torch.Tensor):
                result.requires_grad_()
            else:
                for r in result:
                    r.requires_grad_()

            assert self.model.pipeline
            return PipeBackRedirect.apply(
                result, dest_global_rank, grads, self.model.pipeline.transport, futures, self.model
            )

    @property
    def final_stage(self) -> bool:
        return self.model.final_stage

    def _recv_result(self, shapes: SizeOrSizes, dtypes: DtypeOrDtypes, message: PipeMessage) -> TensorOrTensors:
        group = get_pipeline_parallel_group()
        set_device_based_on_group(group)

        assert self.model.pipeline
        transport = self.model.pipeline.transport

        if isinstance(shapes, torch.Size):
            message.tensor_shapes = [cast(torch.Size, shapes)]
            message.tensor_dtypes = [cast(torch.dtype, dtypes)]
            message = transport.recv_message_tensors(message)
            return message.tensors[0]
        else:
            message.tensor_shapes = cast(List[torch.Size], shapes)
            message.tensor_dtypes = cast(List[torch.dtype], dtypes)
            message = transport.recv_message_tensors(message)
            return message.tensors

    @staticmethod
    def _send_result_and_do_backwards(training: bool, message: PipeMessage, grads_message: PipeMessage) -> None:
        try:
            group = get_pipeline_parallel_group()
            set_device_based_on_group(group)
            result = PipeResult
            model = PipeModel

            if isinstance(result, torch.Tensor):
                result = tuple([result])

            message.tensors = tuple(result)
            assert model.pipeline
            transport = model.pipeline.transport
            transport.send_message(message, sync=False, skip_header=True)

            if training:
                grads_message.tensor_shapes = [r.shape for r in result]
                grads_message.tensor_dtypes = [r.dtype for r in result]
                grads_message = transport.recv_message_tensors(grads_message)

                with model.lock:
                    torch.autograd.backward(result, grads_message.tensors, retain_graph=True)
        except Exception as e:
            print(f"Exception raised in rpc callback: {e}")
            raise e

    @staticmethod
    def _register_remote_model(args: List[Any], kwargs: Dict[str, Any]) -> None:
        group = get_pipeline_parallel_group()  # FIXME(tom) handle dynamic group
        set_device_based_on_group(group)
        kwargs["group"] = group
        kwargs["input_device"] = torch.device("cuda", torch.cuda.current_device())
        model = Pipe(*args, **kwargs)
        model.cuda()
        global PipeModel
        PipeModel = model

    @staticmethod
    def _model_forward(
        training: bool, shape: torch.Size, dtype: torch.dtype
    ) -> Optional[Tuple[SizeOrSizes, DtypeOrDtypes]]:
        try:
            model = PipeModel
            assert model.group
            set_device_based_on_group(model.group)

            if isinstance(shape, torch.Size):
                tensor = torch.empty(shape, dtype=dtype)
            else:
                tensor = tuple([torch.empty(s, dtype=d) for s, d in zip(shape, dtype)])

            model.train(training)
            result = model(tensor)
            if model.final_stage:
                global PipeResult
                PipeResult = result
                return (get_shapes(result), get_dtype(result))

            return None
        except Exception as e:
            print(f"Exception raised in rpc callback: {e}")
            raise e
