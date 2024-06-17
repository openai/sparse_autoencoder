# bare bones training script using sparse kernels and sharding/data parallel.
# the main purpose of this code is to provide a reference implementation to compare
# against when implementing our training methodology into other codebases, and to
# demonstrate how sharding/DP can be implemented for autoencoders. some limitations:
# - many basic features (e.g checkpointing, data loading, validation) are not implemented,
# - the codebase is not designed to be extensible or easily hackable.
# - this code is not guaranteed to run efficiently out of the box / in
#   combination with other changes, so you should profile it and make changes as needed.
#
# example launch command:
#    torchrun --nproc-per-node 8 train.py


import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from sparse_autoencoder.kernels import *
from torch.distributed import ReduceOp

RANK = int(os.environ.get("RANK", "0"))


## parallelism


@dataclass
class Comm:
    group: torch.distributed.ProcessGroup

    def all_reduce(self, x, op=ReduceOp.SUM, async_op=False):
        return dist.all_reduce(x, op=op, group=self.group, async_op=async_op)

    def all_gather(self, x_list, x, async_op=False):
        return dist.all_gather(list(x_list), x, group=self.group, async_op=async_op)

    def broadcast(self, x, src, async_op=False):
        return dist.broadcast(x, src, group=self.group, async_op=async_op)

    def barrier(self):
        return dist.barrier(group=self.group)

    def size(self):
        return self.group.size()


@dataclass
class ShardingComms:
    n_replicas: int
    n_op_shards: int
    dp_rank: int
    sh_rank: int
    dp_comm: Comm | None
    sh_comm: Comm | None
    _rank: int

    def sh_allreduce_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sh_comm is None:
            return x

        class AllreduceForward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                assert self.sh_comm is not None
                self.sh_comm.all_reduce(input, async_op=True)
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        return AllreduceForward.apply(x)  # type: ignore

    def sh_allreduce_backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sh_comm is None:
            return x

        class AllreduceBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                grad_output = grad_output.clone()
                assert self.sh_comm is not None
                self.sh_comm.all_reduce(grad_output, async_op=True)
                return grad_output

        return AllreduceBackward.apply(x)  # type: ignore

    def init_broadcast_(self, autoencoder):
        if self.dp_comm is not None:
            for p in autoencoder.parameters():
                self.dp_comm.broadcast(
                    maybe_transpose(p.data),
                    replica_shard_to_rank(
                        replica_idx=0,
                        shard_idx=self.sh_rank,
                        n_op_shards=self.n_op_shards,
                    ),
                )
        
        if self.sh_comm is not None:
            # pre_bias is the same across all shards
            self.sh_comm.broadcast(
                autoencoder.pre_bias.data,
                replica_shard_to_rank(
                    replica_idx=self.dp_rank,
                    shard_idx=0,
                    n_op_shards=self.n_op_shards,
                ),
            )

    def dp_allreduce_(self, autoencoder) -> None:
        if self.dp_comm is None:
            return

        for param in autoencoder.parameters():
            if param.grad is not None:
                self.dp_comm.all_reduce(maybe_transpose(param.grad), op=ReduceOp.AVG, async_op=True)

        # make sure statistics for dead neurons are correct
        self.dp_comm.all_reduce(  # type: ignore
            autoencoder.stats_last_nonzero, op=ReduceOp.MIN, async_op=True
        )

    def sh_allreduce_scale(self, scaler):
        if self.sh_comm is None:
            return

        if hasattr(scaler, "_scale") and scaler._scale is not None:
            self.sh_comm.all_reduce(scaler._scale, op=ReduceOp.MIN, async_op=True)
            self.sh_comm.all_reduce(scaler._growth_tracker, op=ReduceOp.MIN, async_op=True)

    def _sh_comm_op(self, x, op):
        if isinstance(x, (float, int)):
            x = torch.tensor(x, device="cuda")

        if not x.is_cuda:
            x = x.cuda()

        if self.sh_comm is None:
            return x

        out = x.clone()
        self.sh_comm.all_reduce(x, op=op, async_op=True)
        return out

    def sh_sum(self, x: torch.Tensor) -> torch.Tensor:
        return self._sh_comm_op(x, ReduceOp.SUM)

    def all_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        if self.dp_comm is not None:
            self.dp_comm.broadcast(
                x,
                replica_shard_to_rank(
                    replica_idx=0,
                    shard_idx=self.sh_rank,
                    n_op_shards=self.n_op_shards,
                ),
            )

        if self.sh_comm is not None:
            self.sh_comm.broadcast(
                x,
                replica_shard_to_rank(
                    replica_idx=self.dp_rank,
                    shard_idx=0,
                    n_op_shards=self.n_op_shards,
                ),
            )

        return x


def make_torch_comms(n_op_shards=4, n_replicas=2):
    if "RANK" not in os.environ:
        assert n_op_shards == 1
        assert n_replicas == 1
        return TRIVIAL_COMMS

    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)

    print(f"{rank=}, {world_size=}")
    dist.init_process_group("nccl")

    my_op_shard_idx = rank % n_op_shards
    my_replica_idx = rank // n_op_shards

    shard_rank_lists = [list(range(i, i + n_op_shards)) for i in range(0, world_size, n_op_shards)]

    shard_groups = [dist.new_group(shard_rank_list) for shard_rank_list in shard_rank_lists]

    my_shard_group = shard_groups[my_replica_idx]

    replica_rank_lists = [
        list(range(i, n_op_shards * n_replicas, n_op_shards)) for i in range(n_op_shards)
    ]

    replica_groups = [dist.new_group(replica_rank_list) for replica_rank_list in replica_rank_lists]

    my_replica_group = replica_groups[my_op_shard_idx]

    torch.distributed.all_reduce(torch.ones(1).cuda())
    torch.cuda.synchronize()

    dp_comm = Comm(group=my_replica_group)
    sh_comm = Comm(group=my_shard_group)

    return ShardingComms(
        n_replicas=n_replicas,
        n_op_shards=n_op_shards,
        dp_comm=dp_comm,
        sh_comm=sh_comm,
        dp_rank=my_replica_idx,
        sh_rank=my_op_shard_idx,
        _rank=rank,
    )


def replica_shard_to_rank(replica_idx, shard_idx, n_op_shards):
    return replica_idx * n_op_shards + shard_idx


TRIVIAL_COMMS = ShardingComms(
    n_replicas=1,
    n_op_shards=1,
    dp_rank=0,
    sh_rank=0,
    dp_comm=None,
    sh_comm=None,
    _rank=0,
)


def sharded_topk(x, k, sh_comm, capacity_factor=None):
    batch = x.shape[0]

    if capacity_factor is not None:
        k_in = min(int(k * capacity_factor // sh_comm.size()), k)
    else:
        k_in = k

    topk = torch.topk(x, k=k_in, dim=-1)
    inds = topk.indices
    vals = topk.values

    if sh_comm is None:
        return inds, vals

    all_vals = torch.empty(sh_comm.size(), batch, k_in, dtype=vals.dtype, device=vals.device)
    sh_comm.all_gather(all_vals, vals, async_op=True)

    all_vals = all_vals.permute(1, 0, 2)  # put shard dim next to k
    all_vals = all_vals.reshape(batch, -1)  # flatten shard into k

    all_topk = torch.topk(all_vals, k=k, dim=-1)
    global_topk = all_topk.values

    dummy_vals = torch.zeros_like(vals)
    dummy_inds = torch.zeros_like(inds)

    my_inds = torch.where(vals >= global_topk[:, [-1]], inds, dummy_inds)
    my_vals = torch.where(vals >= global_topk[:, [-1]], vals, dummy_vals)

    return my_inds, my_vals


## autoencoder


class FastAutoencoder(nn.Module):
    """
    Top-K Autoencoder with sparse kernels. Implements:

        latents = relu(topk(encoder(x - pre_bias) + latent_bias))
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        n_dirs_local: int,
        d_model: int,
        k: int,
        auxk: int | None,
        dead_steps_threshold: int,
        comms: ShardingComms | None = None,
    ):
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.comms = comms if comms is not None else TRIVIAL_COMMS
        self.dead_steps_threshold = dead_steps_threshold

        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))

        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dirs_local, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_steps_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

        ## initialization

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

        # store decoder in column major layout for kernel
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T

        unit_norm_decoder_(self)

    @property
    def n_dirs(self):
        return self.n_dirs_local * self.comms.n_op_shards

    def forward(self, x):
        class EncWrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, pre_bias, weight, latent_bias):
                x = x - pre_bias
                latents_pre_act = F.linear(x, weight, latent_bias)

                inds, vals = sharded_topk(
                    latents_pre_act,
                    k=self.k,
                    sh_comm=self.comms.sh_comm,
                    capacity_factor=4,
                )

                ## set num nonzero stat ##
                tmp = torch.zeros_like(self.stats_last_nonzero)
                tmp.scatter_add_(
                    0,
                    inds.reshape(-1),
                    (vals > 1e-3).to(tmp.dtype).reshape(-1),
                )
                self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
                self.stats_last_nonzero += 1
                ## end stats ##

                ## auxk
                if self.auxk is not None:  # for auxk
                    # IMPORTANT: has to go after stats update!
                    # WARN: auxk_mask_fn can mutate latents_pre_act!
                    auxk_inds, auxk_vals = sharded_topk(
                        self.auxk_mask_fn(latents_pre_act),
                        k=self.auxk,
                        sh_comm=self.comms.sh_comm,
                        capacity_factor=2,
                    )
                    ctx.save_for_backward(x, weight, inds, auxk_inds)
                else:
                    ctx.save_for_backward(x, weight, inds)
                    auxk_inds = None
                    auxk_vals = None

                ## end auxk

                return (
                    inds,
                    vals,
                    auxk_inds,
                    auxk_vals,
                )

            @staticmethod
            def backward(ctx, _, grad_vals, __, grad_auxk_vals):
                # encoder backwards
                if self.auxk is not None:
                    x, weight, inds, auxk_inds = ctx.saved_tensors

                    all_inds = torch.cat((inds, auxk_inds), dim=-1)
                    all_grad_vals = torch.cat((grad_vals, grad_auxk_vals), dim=-1)
                else:
                    x, weight, inds = ctx.saved_tensors

                    all_inds = inds
                    all_grad_vals = grad_vals

                grad_sum = torch.zeros(self.n_dirs_local, dtype=torch.float32, device=grad_vals.device)
                grad_sum.scatter_add_(
                    -1, all_inds.flatten(), all_grad_vals.flatten().to(torch.float32)
                )

                return (
                    None,
                    # pre_bias grad optimization - can reduce before mat-vec multiply
                    -(grad_sum @ weight),
                    triton_sparse_transpose_dense_matmul(all_inds, all_grad_vals, x, N=self.n_dirs_local),
                    grad_sum,
                )

        pre_bias = self.comms.sh_allreduce_backward(self.pre_bias)

        # encoder
        inds, vals, auxk_inds, auxk_vals = EncWrapper.apply(
            x, pre_bias, self.encoder.weight, self.latent_bias
        )

        vals = torch.relu(vals)
        if auxk_vals is not None:
            auxk_vals = torch.relu(auxk_vals)

        recons = self.decode_sparse(inds, vals)

        return recons, {
            "auxk_inds": auxk_inds,
            "auxk_vals": auxk_vals,
        }

    def decode_sparse(self, inds, vals):
        recons = TritonDecoderAutograd.apply(inds, vals, self.decoder.weight)
        recons = self.comms.sh_allreduce_forward(recons)

        return recons + self.pre_bias


def unit_norm_decoder_(autoencoder: FastAutoencoder) -> None:
    """
    Unit normalize the decoder weights of an autoencoder.
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)


def unit_norm_decoder_grad_adjustment_(autoencoder) -> None:
    """project out gradient information parallel to the dictionary vectors - assumes that the decoder is already unit normed"""

    assert autoencoder.decoder.weight.grad is not None

    triton_add_mul_(
        autoencoder.decoder.weight.grad,
        torch.einsum("bn,bn->n", autoencoder.decoder.weight.data, autoencoder.decoder.weight.grad),
        autoencoder.decoder.weight.data,
        c=-1,
    )


def maybe_transpose(x):
    return x.T if not x.is_contiguous() and x.T.is_contiguous() else x


def sharded_grad_norm(autoencoder, comms, exclude=None):
    if exclude is None:
        exclude = []
    total_sq_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    exclude = set(exclude)

    total_num_params = 0
    for param in autoencoder.parameters():
        if param in exclude:
            continue
        if param.grad is not None:
            sq_norm = ((param.grad).float() ** 2).sum()
            if param is autoencoder.pre_bias:
                total_sq_norm += sq_norm  # pre_bias is the same across all shards
            else:
                total_sq_norm += comms.sh_sum(sq_norm)

            param_shards = comms.n_op_shards if param is autoencoder.pre_bias else 1
            total_num_params += param.numel() * param_shards

    return total_sq_norm.sqrt()


def batch_tensors(
    it: Iterable[torch.Tensor],
    batch_size: int,
    drop_last=True,
    stream=None,
) -> Iterator[torch.Tensor]:
    """
    input is iterable of tensors of shape [batch_old, ...]
    output is iterable of tensors of shape [batch_size, ...]
    batch_old does not need to be divisible by batch_size
    """

    tensors = []
    batch_so_far = 0

    for t in it:
        tensors.append(t)
        batch_so_far += t.shape[0]

        if sum(t.shape[0] for t in tensors) < batch_size:
            continue

        while batch_so_far >= batch_size:
            if len(tensors) == 1:
                (concat,) = tensors
            else:
                with torch.cuda.stream(stream):
                    concat = torch.cat(tensors, dim=0)

            offset = 0
            while offset + batch_size <= concat.shape[0]:
                yield concat[offset : offset + batch_size]
                batch_so_far -= batch_size
                offset += batch_size

            tensors = [concat[offset:]] if offset < concat.shape[0] else []

    if len(tensors) > 0 and not drop_last:
        yield torch.cat(tensors, dim=0)


def print0(*a, **k):
    if RANK == 0:
        print(*a, **k)


import wandb


class Logger:
    def __init__(self, **kws):
        self.vals = {}
        self.enabled = (RANK == 0) and not kws.pop("dummy", False)
        if self.enabled:
            wandb.init(
                **kws
            )

    def logkv(self, k, v):
        if self.enabled:
            self.vals[k] = v.detach() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self):
        if self.enabled:
            wandb.log(self.vals)
            self.vals = {}


def training_loop_(
    ae, train_acts_iter, loss_fn, lr, comms, eps=6.25e-10, clip_grad=None, ema_multiplier=0.999, logger=None
):
    if logger is None:
        logger = Logger(dummy=True)

    scaler = torch.cuda.amp.GradScaler()
    autocast_ctx_manager = torch.cuda.amp.autocast()

    opt = torch.optim.Adam(ae.parameters(), lr=lr, eps=eps, fused=True)
    if ema_multiplier is not None:
        ema = EmaModel(ae, ema_multiplier=ema_multiplier)

    for i, flat_acts_train_batch in enumerate(train_acts_iter):
        flat_acts_train_batch = flat_acts_train_batch.cuda()

        with autocast_ctx_manager:
            recons, info = ae(flat_acts_train_batch)

            loss = loss_fn(ae, flat_acts_train_batch, recons, info, logger)

        print0(i, loss)

        logger.logkv("loss_scale", scaler.get_scale())

        if RANK == 0:
            wandb.log({"train_loss": loss.item()})

        loss = scaler.scale(loss)
        loss.backward()

        unit_norm_decoder_(ae)
        unit_norm_decoder_grad_adjustment_(ae)

        # allreduce gradients
        comms.dp_allreduce_(ae)

        # keep fp16 loss scale synchronized across shards
        comms.sh_allreduce_scale(scaler)

        # if you want to do anything with the gradients that depends on the absolute scale (e.g clipping, do it after the unscale_)
        scaler.unscale_(opt)

        # gradient clipping
        if clip_grad is not None:
            grad_norm = sharded_grad_norm(ae, comms)
            logger.logkv("grad_norm", grad_norm)
            grads = [x.grad for x in ae.parameters() if x.grad is not None]
            torch._foreach_mul_(grads, clip_grad / torch.clamp(grad_norm, min=clip_grad))

        if ema_multiplier is not None:
            ema.step()

        # take step with optimizer
        scaler.step(opt)
        scaler.update()
        
        logger.dumpkvs()


def init_from_data_(ae, stats_acts_sample, comms):
    from geom_median.torch import compute_geometric_median

    ae.pre_bias.data = (
        compute_geometric_median(stats_acts_sample[:32768].float().cpu()).median.cuda().float()
    )
    comms.all_broadcast(ae.pre_bias.data)

    # encoder initialization (note: in our ablations we couldn't find clear evidence that this is beneficial, this is just to ensure exact match with internal codebase)
    d_model = ae.d_model
    with torch.no_grad():
        x = torch.randn(256, d_model).cuda().to(stats_acts_sample.dtype)
        x /= x.norm(dim=-1, keepdim=True)
        x += ae.pre_bias.data
        comms.all_broadcast(x)
        recons, _ = ae(x)
        recons_norm = (recons - ae.pre_bias.data).norm(dim=-1).mean()

        ae.encoder.weight.data /= recons_norm.item()
        print0("x norm", x.norm(dim=-1).mean().item())
        print0("out norm", (ae(x)[0] - ae.pre_bias.data).norm(dim=-1).mean().item())


from contextlib import contextmanager


@contextmanager
def temporary_weight_swap(model: torch.nn.Module, new_weights: list[torch.Tensor]):
    for _p, new_p in zip(model.parameters(), new_weights, strict=True):
        assert _p.shape == new_p.shape
        _p.data, new_p.data = new_p.data, _p.data

    yield

    for _p, new_p in zip(model.parameters(), new_weights, strict=True):
        assert _p.shape == new_p.shape
        _p.data, new_p.data = new_p.data, _p.data


class EmaModel:
    def __init__(self, model, ema_multiplier):
        self.model = model
        self.ema_multiplier = ema_multiplier
        self.ema_weights = [torch.zeros_like(x, requires_grad=False) for x in model.parameters()]
        self.ema_steps = 0

    def step(self):
        torch._foreach_lerp_(
            self.ema_weights,
            list(self.model.parameters()),
            1 - self.ema_multiplier,
        )
        self.ema_steps += 1

    # context manager for setting the autoencoder weights to the EMA weights
    @contextmanager
    def use_ema_weights(self):
        assert self.ema_steps > 0

        # apply bias correction
        bias_correction = 1 - self.ema_multiplier**self.ema_steps
        ema_weights_bias_corrected = torch._foreach_div(self.ema_weights, bias_correction)

        with torch.no_grad():
            with temporary_weight_swap(self.model, ema_weights_bias_corrected):
                yield


@dataclass
class Config:
    n_op_shards: int = 1
    n_replicas: int = 8

    n_dirs: int = 32768
    bs: int = 131072
    d_model: int = 768
    k: int = 32
    auxk: int = 256

    lr: float = 1e-4
    eps: float = 6.25e-10
    clip_grad: float | None = None
    auxk_coef: float = 1 / 32
    dead_toks_threshold: int = 10_000_000
    ema_multiplier: float | None = None
    
    wandb_project: str | None = None
    wandb_name: str | None = None


def main():
    cfg = Config()
    comms = make_torch_comms(n_op_shards=cfg.n_op_shards, n_replicas=cfg.n_replicas)

    ## dataloading is left as an exercise for the reader
    acts_iter = ...
    stats_acts_sample = ...

    n_dirs_local = cfg.n_dirs // cfg.n_op_shards
    bs_local = cfg.bs // cfg.n_replicas

    ae = FastAutoencoder(
        n_dirs_local=n_dirs_local,
        d_model=cfg.d_model,
        k=cfg.k,
        auxk=cfg.auxk,
        dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
        comms=comms,
    )
    ae.cuda()
    init_from_data_(ae, stats_acts_sample, comms)
    # IMPORTANT: make sure all DP ranks have the same params
    comms.init_broadcast_(ae)

    mse_scale = (
        1 / ((stats_acts_sample.float().mean(dim=0) - stats_acts_sample.float()) ** 2).mean()
    )
    comms.all_broadcast(mse_scale)
    mse_scale = mse_scale.item()

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=cfg.wandb_project is None,
    )

    training_loop_(
        ae,
        batch_tensors(
            acts_iter,
            bs_local,
            drop_last=True,
        ),
        lambda ae, flat_acts_train_batch, recons, info, logger: (
            # MSE
            logger.logkv("train_recons", mse_scale * mse(recons, flat_acts_train_batch))
            # AuxK
            + logger.logkv(
                "train_maxk_recons",
                cfg.auxk_coef
                * normalized_mse(
                    ae.decode_sparse(
                        info["auxk_inds"],
                        info["auxk_vals"],
                    ),
                    flat_acts_train_batch - recons.detach() + ae.pre_bias.detach(),
                ).nan_to_num(0),
            )
        ),
        lr=cfg.lr,
        eps=cfg.eps,
        clip_grad=cfg.clip_grad,
        ema_multiplier=cfg.ema_multiplier,
        logger=logger,
        comms=comms,
    )


if __name__ == "__main__":
    main()
