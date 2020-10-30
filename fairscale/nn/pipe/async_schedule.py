# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple, Any

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from fairscale.nn.model_parallel import (
    get_pipeline_parallel_ranks,
    get_model_parallel_prev_next_ranks,
    get_model_parallel_group,
)

from torch.distributed.distributed_c10d import _get_global_rank

from .messages import Transport, MESSAGE_TENSOR_SIZE
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals
from .types import EVENT_LOOP_QUEUE, PipelineStyle, PipeMessage, Tensors

import os

from inspect import currentframe, getframeinfo

cactus = open("/private/home/tbirch/src/fairscale/debug-" + os.environ["OMPI_COMM_WORLD_RANK"], "w", buffering=1)


@dataclass(frozen=True)
class Location:
    stage: int
    index: int

    def __repr__(self) -> str:
        return f"{self.stage}@{self.index}"


@dataclass(frozen=True)
class Invocation:
    order: int
    this: Location
    source: Optional[Location]
    dest: Optional[Location]


Activations = Dict[int, Dict[int, Dict[int, Batch]]]
Invocations = Dict[int, Invocation]


ARO_INDICES = []

G_WORK_ITEM = None


@dataclass(frozen=True)
class TailBackwardContext:
    activations: Activations
    invocations: Invocations
    count_per_order: Dict[int, int]
    expected_gradients: int


class ModuleWrapper:
    def __init__(self, module: nn.Sequential, location: Location, invocations: Optional[List[Invocation]] = None):
        self.module: nn.Sequential = module
        self.location: Location = location
        self.invocations: List[Invocation] = invocations or []

    def __repr__(self) -> str:
        return f"{self.location}:\n" + "\n".join(map(str, self.invocations)) + "\n\t" + str(self.module)

    def __len__(self) -> int:
        return len(self.module)

    def __iter__(self) -> Iterable:
        yield from self.module


class AsyncMessageType(Enum):
    Activations = auto()
    Gradients = auto()


@dataclass(frozen=True)
class AsyncMessageBody:
    message_type: AsyncMessageType
    microbatch_index: int
    source: Location
    dest: Location
    order: int


class AutogradWithoutActivations(torch.autograd.Function):
    """A helper class to add another edge in the autograd graph which allows us
    to delete the potentially large activations and still perform a backward
    pass. Returns return a phony tensor which is connected to the graph."""

    @staticmethod
    # type: ignore
    def forward(ctx, *x):
        return torch.tensor(1.0)

    @staticmethod
    # type: ignore
    def backward(ctx, grad):
        assert ctx.grad_from_pipeline is not None
        return ctx.grad_from_pipeline


class AsyncRecvOperator(torch.autograd.Function):
    """Receive activations to the previous pipeline stage"""

    @staticmethod
    # type: ignore
    def forward(ctx, phony: Tensor, transport: Transport, message: PipeMessage) -> Tensors:
        ctx.transport = transport
        ctx.index = message.args.microbatch_index

        result = transport.recv_message_tensors(message)

        ctx.args = result.args

        def maybe_requires_grad(t: Tensor) -> Tensor:
            if t.dtype.is_floating_point:
                return t.requires_grad_()
            return t

        # global ARO_INDICES
        # ARO_INDICES = []

        return tuple(maybe_requires_grad(r) for r in result.tensors)

    @staticmethod
    # type: ignore
    def backward(ctx, *grad: Tensor,) -> Tuple[Optional[Tensor], ...]:
        ranks = get_pipeline_parallel_ranks()
        this_rank = torch.distributed.get_rank()
        body = AsyncMessageBody(
            AsyncMessageType.Gradients, ctx.index, source=ctx.args.dest, dest=ctx.args.source, order=ctx.args.order - 1
        )

        # global ARO_INDICES
        # ARO_INDICES.append(ctx.index)
        ctx.transport.send_message(
            PipeMessage(
                this_rank, ranks[ctx.args.source.stage], queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple(grad),
            ),
            sync=True,
        )

        try:
            tail_ctx = getattr(ctx, "tail_ctx", None)
            if tail_ctx:
                expected_gradients = tail_ctx.expected_gradients
                print(f"tail ctx expected_gradients {expected_gradients}")
                while expected_gradients > 0:
                    message = ctx.transport.recv_message_header(EVENT_LOOP_QUEUE)

                    args: AsyncMessageBody = message.args
                    assert args.message_type is AsyncMessageType.Gradients

                    invocation = tail_ctx.invocations[args.order]
                    expected_gradients -= tail_ctx.count_per_order[invocation.order]
                    recvd_grads = ctx.transport.recv_message_tensors(message)

                    AsyncEventLoop.perform_backward_for_invocation(
                        ctx.transport, message, recvd_grads, tail_ctx.activations, invocation
                    )
        except Exception as e:
            print(f"fucaljkrlka {e}")

        # if len(ARO_INDICES) == 10:
        #    print(f"aro done {this_rank}")

        return (None, None, None, None, None)


class AsyncEventLoop:
    def __init__(
        self,
        partitions: List[ModuleWrapper],
        group: ProcessGroup,
        transport: Transport,
        training: bool,
        checkpoint_stop: int,
    ):
        self.training = training
        self.checkpoint_stop = checkpoint_stop
        self.transport = transport
        self.group = group
        self.partitions: List[ModuleWrapper] = partitions
        self.broadcast_stream = torch.cuda.Stream()

    def send_async_message(self, dst_rank: int, result: Batch, invocation: Invocation) -> Batch:
        """Send batch to dst_rank, and use AutogradWithoutActivations to delete
        the activations since we no longer need them"""

        assert invocation.dest
        src_rank = torch.distributed.get_rank()

        body = AsyncMessageBody(
            AsyncMessageType.Activations,
            result.index,
            source=invocation.this,
            dest=invocation.dest,
            order=invocation.order + 1,
        )

        self.transport.send_message(
            PipeMessage(src_rank, dst_rank, queue_name=EVENT_LOOP_QUEUE, args=body, tensors=tuple([*result])),
            sync=True,
        )

        phony = AutogradWithoutActivations.apply(*result)
        return Batch(phony, result.index)

    def run_invocation(
        self,
        batch: Batch,
        partition: ModuleWrapper,
        skip_trackers: List[SkipTrackerThroughPotals],
        invocation: Invocation,
    ) -> Batch:
        """Actually run the forward pass for a given module, and send the result
        to the next stage in the pipeline if needed."""
        assert self.group
        from .pipeline import create_task

        task = create_task(
            PipelineStyle.AsyncSchedule,
            self.checkpoint_stop,
            batch.index,
            self.group.rank(),
            batch,
            partition.module,
            skip_trackers,
            [],
        )
        result = task.compute()
        task.finalize(result)

        if invocation.dest and invocation.dest.stage != invocation.this.stage:
            ranks = get_pipeline_parallel_ranks()
            dst_rank = ranks[invocation.dest.stage]
            result = self.send_async_message(dst_rank, result, invocation)
        return result

    @staticmethod
    def perform_backward_for_invocation(
        transport, message: PipeMessage, recvd_grads, activations: Activations, invocation: Invocation
    ) -> None:
        """Perform the backward pass by looking up the appropriate `Batch` and
        then calling `backward` on the tensor"""

        # recvd_grads = transport.recv_message_tensors(message)
        print(f"{invocation.this.index}, {invocation.order}, {message.args.microbatch_index}", file=cactus)

        batch: Batch = activations[invocation.this.index][invocation.order][message.args.microbatch_index]

        # All batches saved in `activations` are generated by AutogradWithoutActivations,
        # so we store the gradients in `grad_from_pipeline` so it will be used
        # during the backward pass
        batch.tensor.grad_fn.grad_from_pipeline = tuple(recvd_grads.tensors)  # type: ignore
        batch.tensor.backward(retain_graph=True)

    def run_invocations_on_batch(
        self,
        batch: Batch,
        invocations: Invocations,
        order: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
    ) -> Tuple[int, int]:
        """Run invocations on the batch until we hit one that receives its input
        from a different stage (i.e. another process)"""

        invocations_handled = 0
        last_order = 0
        for invocation in invocations.values():
            if invocation.order < order:
                continue
            pi = invocation.this.index
            partition = self.partitions[pi]

            if invocation.order == order:
                invocations_handled += 1
                last_order = invocation.order
                print(f">>> forward {pi}, {invocation.order}, {batch.index}", file=cactus)
                activations[pi][invocation.order][batch.index] = self.run_invocation(
                    batch, partition, skip_trackers, invocation
                )
                print(f"<<< forward {pi}, {invocation.order}, {batch.index}", file=cactus)
            elif invocation.source and invocation.source.stage == self.group.rank():
                invocations_handled += 1
                last_order = invocation.order
                batch = activations[invocation.source.index][invocation.order - 1][batch.index]
                print(f">>> forward {pi}, {invocation.order}, {batch.index}", file=cactus)
                activations[pi][invocation.order][batch.index] = self.run_invocation(
                    batch, partition, skip_trackers, invocation
                )
                print(f"<<< forward {pi}, {invocation.order}, {batch.index}", file=cactus)
                del activations[invocation.source.index][invocation.order - 1][batch.index]

            elif invocation.source and invocation.source.stage != self.group.rank():
                break

        return (invocations_handled, last_order)

    def event_loop_head(
        self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals], event: Optional[Event]
    ) -> Dict[str, Any]:
        """The event loop for the "head", which first performs the forward pass
        on any applicable layers for this stage, and then enters the common
        `event_loop_inner`"""

        invocations, activations = self.get_invocations_and_activations()

        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0

        count_per_order = dict()

        if actual_invocations < expected_invocations:
            self.event_loop_inner(
                expected_invocations,
                skip_trackers,
                activations,
                invocations,
                count_per_order,
                already_received=actual_invocations,
                event=event,
                ignore_gradients=True,
                batches=batches,
            )

        if self.training:
            return {
                "activations": activations,
                "invocations": invocations,
                "count_per_order": count_per_order,
                "batches": len(batches),
                "skip_trackers": skip_trackers,
            }
        else:
            return {}

    def head_backwards(self, head_ctx: Dict[str, Any]):
        invocations: Invocations = head_ctx["invocations"]
        activations: Activations = head_ctx["activations"]
        batches: int = head_ctx["batches"]
        expected_invocations = len(invocations) * batches
        assert self.training
        self.event_loop_inner(
            expected_invocations,
            head_ctx["skip_trackers"],
            activations,
            invocations,
            head_ctx["count_per_order"],
            already_received=expected_invocations,
        )

    def get_batch_from_message(self, message: PipeMessage) -> Batch:
        """Get the tensor(s) wrapped in a `Batch` from a `PipeMessage`, applying
        AsyncRecvOperator so we can intercept the backward pass"""

        microbatch_index = message.args.microbatch_index
        phony = torch.empty(0, device=self.transport.input_device, requires_grad=True)
        result = AsyncRecvOperator.apply(phony, self.transport, message)
        if len(result) == 1:
            batch = Batch(result[0], microbatch_index)
        else:
            batch = Batch(result, microbatch_index)
        return batch

    def event_loop_tail(self, batches: List[Batch], skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        """The event loop for the "tail", or final stage which only processes
        activations and then returns to the caller so that the loss can be
        calculated. This also handles the first/only stage for the special
        case of a 1-stage pipeline."""

        assert self.group

        invocations, activations = self.get_invocations_and_activations()
        expected_invocations = len(invocations) * len(batches)
        actual_invocations = 0

        rank = self.group.rank()
        count_per_order = dict()

        for batchi, batch in enumerate(batches):
            if rank == 0:
                order = 0
            else:
                message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
                args: AsyncMessageBody = message.args

                batch = self.get_batch_from_message(message)
                order = args.order

            print(f"tail-eli {batchi}, {len(batches)}, {torch.distributed.get_rank()}", file=cactus)

            inv_count, last_order = self.run_invocations_on_batch(batch, invocations, order, skip_trackers, activations)
            actual_invocations += inv_count
            count_per_order[last_order] = inv_count

            try:
                if invocations[last_order].dest is None:
                    self.prepare_tail_backward(
                        batch, activations, invocations, count_per_order, len(invocations) - inv_count
                    )
            except Exception as e:
                print(f"fucaljkrlka {e}")

        if actual_invocations < expected_invocations:
            expected_gradients = 0  # (len(invocations) - 1) * len(batches)

            self.event_loop_inner(
                expected_invocations,
                skip_trackers,
                activations,
                invocations,
                count_per_order,
                already_received=actual_invocations,
                ignore_gradients=True,
                tail=True,
            )

        for index, batch in activations[len(self.partitions) - 1][next(reversed(invocations.values())).order].items():
            batches[index] = batch

    def get_invocations_and_activations(self) -> Tuple[Invocations, Activations]:
        activations: Activations = dict()
        invocations: Invocations = OrderedDict()

        for pi, partition in enumerate(self.partitions):
            activations[pi] = dict()
            for invocation in partition.invocations:
                activations[pi][invocation.order] = dict()
                invocations[invocation.order] = invocation

        invocations = OrderedDict(sorted(invocations.items(), key=lambda entry: entry[0]))

        return (invocations, activations)

    def event_loop(self, num_microbatch: int, skip_trackers: List[SkipTrackerThroughPotals]) -> None:
        """The event loop for the "middle", i.e. neither the head nor the tail"""
        assert self.group

        invocations, activations = self.get_invocations_and_activations()

        expected_invocations = len(invocations) * num_microbatch

        self.event_loop_inner(expected_invocations, skip_trackers, activations, invocations, dict())

    def event_loop_inner(
        self,
        expected_invocations: int,
        skip_trackers: List[SkipTrackerThroughPotals],
        activations: Activations,
        invocations: Invocations,
        count_per_order: Dict[int, int],
        *,
        batches: Optional[List[Batch]] = None,
        already_received: int = 0,
        ignore_gradients: bool = False,
        event: Optional[Event] = None,
        tail: bool = False,
    ) -> None:
        """The common event loop shared by all stages. This processses
        activations for the forward pass, and if `self.training` is true,
        processes gradients for the backward pass."""

        num_activations = already_received
        if self.training and not ignore_gradients:
            num_gradients = 0
        else:
            num_gradients = expected_invocations

        stashed_messages = {}
        batch_iter = iter(batches) if batches else None
        num_batches = expected_invocations / len(invocations)

        message_tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=self.transport.input_device)
        gr = get_model_parallel_group()
        should_use_irecv = (
            any(inv.source and inv.source.stage != self.group.rank() for inv in invocations.values()) and gr.rank() == 0
        )

        if should_use_irecv:  # and num_activations < expected_invocations:
            print(f"yay work_itemm")
            work_item = torch.distributed.irecv(message_tensor, src=None, tag=EVENT_LOOP_QUEUE, group=self.group)
        else:
            work_item = None

        work_item_done = False

        if batch_iter:
            processed_batch_count = 0
        else:
            processed_batch_count = num_batches

        inv_per_batch = 0

        argop, _ = get_model_parallel_prev_next_ranks()

        while num_activations < expected_invocations or num_gradients < expected_invocations:
            print(f"eli {num_activations}, {num_gradients}, {expected_invocations}, {processed_batch_count}, {torch.distributed.get_rank()}")
            if num_activations == expected_invocations and num_gradients == 0 and event is not None:
                # We are ready to do the backward pass, but must wait for
                # PipeRPCWrapper to signal that it is safe to proceed, otherwise
                # deadlock
                pass  # event.wait()

            message = None
            inv_order = None
            microbatch_index = None
            batch = None
            message_type = None

            if gr.rank() == 0:
                if work_item and (work_item.is_completed() or processed_batch_count == num_batches):
                    print(f"work item!")
                    work_item.wait()
                    message = self.transport.recv_message_header(EVENT_LOOP_QUEUE, future=message_tensor)
                    args: AsyncMessageBody = message.args
                    message_type = args.message_type
                    inv_order = args.order
                    microbatch_index = args.microbatch_index
                    work_item = None
                elif batch_iter and processed_batch_count < num_batches:
                    batch = next(batch_iter)
                    processed_batch_count += 1
                    inv_order = 0
                    microbatch_index = batch.index
                    message_type = AsyncMessageType.Activations
                else:
                    message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
                    args: AsyncMessageBody = message.args
                    message_type = args.message_type
                    inv_order = args.order
                    microbatch_index = args.microbatch_index

                sequence = torch.tensor([inv_order, microbatch_index]).cuda()
                with torch.cuda.stream(self.broadcast_stream):
                    print(f">>> broadcast", file=cactus)
                    # torch.distributed.broadcast(sequence, src=_get_global_rank(gr, 0), group=gr)
                    for rank in range(1, gr.size()):
                        torch.distributed.send(sequence, _get_global_rank(gr, rank), tag=1, group=argop)
                    torch.cuda.current_stream().synchronize()
                    print(f"<<< broadcast", file=cactus)
                torch.distributed.barrier(group=gr)
            else:
                sequence = torch.tensor([0, 0]).cuda()
                if len(stashed_messages) == 0 and not batch_iter:  # FIXME(tom) predict if next is batch?
                    message = None #self.transport.recv_message_header(EVENT_LOOP_QUEUE)

                with torch.cuda.stream(self.broadcast_stream):
                    print(f">>> broadcast", file=cactus)
                    # torch.distributed.broadcast(sequence, src=_get_global_rank(gr, 0), group=gr)
                    torch.distributed.recv(sequence, _get_global_rank(gr, 0), tag=1, group=argop)
                    torch.cuda.current_stream().synchronize()
                    print(f"<<< broadcast", file=cactus)
                    expected_sequence = tuple(sequence.tolist())
                torch.distributed.barrier(group=gr)

                while True:
                    if message is None:
                        message = stashed_messages.get(expected_sequence, None)

                    if message:
                        args = message.args
                    elif batch_iter and expected_sequence == (0, processed_batch_count):
                        batch = next(batch_iter)
                        processed_batch_count += 1
                        inv_order = 0
                        microbatch_index = batch.index
                        message_type = AsyncMessageType.Activations
                        break
                    else:
                        message = self.transport.recv_message_header(EVENT_LOOP_QUEUE)
                        args: AsyncMessageBody = message.args

                    st = (args.order, args.microbatch_index)
                    if st != expected_sequence:
                        stashed_messages[st] = message
                        message = None
                    else:
                        if st in stashed_messages:
                            del stashed_messages[st]
                        inv_order = args.order
                        microbatch_index = args.microbatch_index
                        message_type = args.message_type
                        break

            # torch.distributed.barrier(group=gr)
            try:
                assert sequence[0] == inv_order
                assert sequence[1] == microbatch_index
            except Exception as e:
                print(f"asssert {sequence}, {[inv_order, args.microbatch_index]}")
                raise e

            invocation = invocations[inv_order]

            # FIXME(tom) for combining pipeline with megatron, I currently don't
            # control the order of received activations or gradients, so it is
            # possible for a reused ColumnParallelLinear for example to receive
            # a different order of activations w.r.t. the sending stage, which
            # would result in incorrect values being used for the all_gather
            if message_type is AsyncMessageType.Activations:
                if batch is None:
                    batch = self.get_batch_from_message(message)

                if work_item and work_item.is_completed():
                    work_item.wait()
                    work_item_done = True

                inv_count, last_order = self.run_invocations_on_batch(
                    batch, invocations, inv_order, skip_trackers, activations
                )
                if inv_order == 0:
                    inv_per_batch = inv_count

                count_per_order[last_order] = inv_count
                num_activations += inv_count
                try:
                    if tail and invocations[last_order].dest is None:
                        self.prepare_tail_backward(
                            batch, activations, invocations, count_per_order, len(invocations) - inv_count
                        )

                    assert num_activations <= expected_invocations
                except Exception as e:
                    print(f"fucaljkrlka {e}")

                if (
                    work_item is None
                    and should_use_irecv
                    and expected_invocations - num_activations > (num_batches - processed_batch_count) * inv_per_batch
                ):
                    work_item = torch.distributed.irecv(
                        message_tensor, src=None, tag=EVENT_LOOP_QUEUE, group=self.group
                    )
                    work_item_done = False

            elif message_type is AsyncMessageType.Gradients:
                num_gradients += count_per_order[invocation.order]

                recvd_grads = self.transport.recv_message_tensors(message)

                if work_item and work_item.is_completed():
                    work_item.wait()
                    work_item_done = True

                self.perform_backward_for_invocation(self.transport, message, recvd_grads, activations, invocation)
                if work_item is None and should_use_irecv and num_gradients < expected_invocations:
                    work_item = torch.distributed.irecv(
                        message_tensor, src=None, tag=EVENT_LOOP_QUEUE, group=self.group
                    )
                    work_item_done = False

    @staticmethod
    def prepare_tail_backward(
        batch: Batch,
        activations: Activations,
        invocations: Invocations,
        count_per_order: Dict[int, int],
        expected_gradients: int,
    ):
        if expected_gradients > 0:
            grad_fn = next(b.grad_fn for b in batch if b.requires_grad)
            grad_fn.tail_ctx = TailBackwardContext(activations, invocations, count_per_order, expected_gradients)
