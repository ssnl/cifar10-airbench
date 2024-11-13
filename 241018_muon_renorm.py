from typing import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import functools
import dataclasses
import math
import time
import random
from distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig  # type: ignore


from typing import Optional
from typing import Protocol
from collections import namedtuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import torch
import functools
import os

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None, dtype=torch.float32, **kwargs):
    U, S, V = G.to(dtype).svd()
    return (U @ V.T).to(G.dtype)

def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7, dtype=torch.bfloat16,
                                 abc: torch.Tensor = torch.tensor((3.4445, -4.7750,  2.0315)), G_fro: Optional[torch.Tensor] = None):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    if G_fro is None:
        G_fro = G.norm()
    denom = G_fro + eps
    X = G.to(dtype) / denom # ensure top singular value <= 1
    abc = abc.expand(steps, 3).to(G.device, dtype)
    # a, b, c = abc
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in zip(*abc.unbind(dim=1)):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_via_newtonschulz5 = functools.wraps(_zeropower_via_newtonschulz5)(torch.compile(_zeropower_via_newtonschulz5))

b_zeropower_via_newtonschulz5 = torch.vmap(
    zeropower_via_newtonschulz5,
    in_dims=(-3,),
)

def make_schedule(iter5, itermid, iter3):
    abcbad = torch.tensor((3.4445, -4.7750,  2.0315))
    abcgood = torch.tensor((1.5, -0.5, 0))
    abcschedule = torch.cat([
        abcbad.expand(iter5, 3),
        torch.lerp(abcbad, abcgood, torch.linspace(0, 1, itermid + 2)[1:-1, None]),
        abcgood.expand(iter3, 3),
    ], 0)
    return abcschedule


zeropower_backends = dict(
    sgd=lambda g, **kwargs: g,
    sign=lambda g, **kwargs: g.sign(),
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    newtonschulz5_proper=functools.partial(zeropower_via_newtonschulz5, abc=torch.tensor((1.5, -0.5, 0))),
    newtonschulz5_sched5=functools.partial(zeropower_via_newtonschulz5, steps=8, abc=make_schedule(3, 1, 1)),
    newtonschulz5_sched8=functools.partial(zeropower_via_newtonschulz5, steps=8, abc=make_schedule(7, 1, 0)),
    newtonschulz5_sched10=functools.partial(zeropower_via_newtonschulz5, steps=10, abc=make_schedule(8, 1, 1)),
    newtonschulz5_sched14=functools.partial(zeropower_via_newtonschulz5, steps=14, abc=make_schedule(10, 2, 2)),
)


@dataclasses.dataclass
class NormInterface:
    state: dict
    grad: torch.Tensor                   # p.grad
    rawg: torch.Tensor                   # possibly momentum-averaged grad
    cond_rawg: torch.Tensor              # possibly preconditioned rawg
    rawg0: torch.Tensor                  # cond_rawg^0
    g: torch.Tensor                      # possibly momentum-averaged rawg^0
    rawgnorm_fro: torch.Tensor           # ||rawg||_fro
    momentum_kind: str
    zeropower_backend: str
    eps: float
    cached_norms: Dict[Tuple[str, str, bool], torch.Tensor] = dataclasses.field(default_factory=dict)

    @property
    def fan_in(self):
        return self.g.size(1)

    @property
    def fan_out(self):
        return self.g.size(0)

    @property
    def min_fan(self):
        return min(self.fan_in, self.fan_out)

    @property
    def max_fan(self):
        return max(self.fan_in, self.fan_out)

    def __call__(self, tensor_kind: str, norm_kind: str, dual: bool = False) -> torch.Tensor:
        cache_key = (tensor_kind, norm_kind, dual)
        if cache_key not in self.cached_norms:
            self.cached_norms[cache_key] = self._compute_norm(tensor_kind, norm_kind, dual)
        return self.cached_norms[cache_key]

    DUAL_NORM_MAP: ClassVar[Dict[str, str]] = dict(
        spectral='nuclear',
        spectral_exact='nuclear_exact',
        nuclear='spectral',
        nuclear_exact='spectral_exact',
        fro='fro',
        fro_exact='fro_exact',
    )

    EXTENDED_NORM_EQUIV_SCALE_MAP: ClassVar[Dict[str, Tuple[str, Callable[[Self], float]]]] = dict(
        # ||W|| := 1 / sqrt(numel) * ||W||_fro
        rms=              ('fro',               lambda self: 1 / self.rawg.numel()**0.5),
        rms_exact=        ('fro_exact',         lambda self: 1 / self.rawg.numel()**0.5),
        # ||W|| := sqrt(fan_in / fan_out) * ||W||_*
        jbnorm=           ('spectral',          lambda self: (self.fan_in / self.fan_out)**0.5),
        jbnorm_exact=     ('spectral_exact',    lambda self: (self.fan_in / self.fan_out)**0.5),
        # ||W|| := sqrt(fan_out / fan_in) * ||W||_*
        jbinvnorm=        ('spectral',          lambda self: (self.fan_out / self.fan_in)**0.5),
        jbinvnorm_exact=  ('spectral_exact',    lambda self: (self.fan_out / self.fan_in)**0.5),
    )

    def _compute_norm(self, tensor_kind: str, norm_kind: str, dual: bool = False):
        scaled_norm_kind, mult_fn = self.EXTENDED_NORM_EQUIV_SCALE_MAP.get(norm_kind, (None, None))
        if scaled_norm_kind is not None:
            mult = mult_fn(self)
            norm = self(tensor_kind, scaled_norm_kind, dual=dual)
            if not dual:
                return norm * mult
            else:
                return norm / mult
        if dual:
            dual_norm_kind = self.DUAL_NORM_MAP[norm_kind]
            if dual_norm_kind is None:
                raise ValueError(f"dual norm kind {norm_kind} not supported")
            return self(tensor_kind, dual_norm_kind, dual=False)
        # else:

        assert tensor_kind in ('rawg', 'g', 'grad')
        if norm_kind in {'fro', 'fro_exact'}:
            if tensor_kind == 'rawg':
                return self.rawgnorm_fro
            elif tensor_kind == 'g':
                if norm_kind == 'fro':
                    assert self.zeropower_backend not in ('svd', 'sign'), "fro norm of g not supported for svd or sign"
                    assert self.momentum_kind not in {'post_ns', 'post_ns_nesterov'}, "fro norm of g not supported for post-ns which breaks g=rawg0"
                    # currently have ||DW||_fro     ~= sqrt(min(fan_in, fan_out))
                    return self.min_fan**0.5
                else:
                    return self.g.norm()
            else:
                assert tensor_kind == 'grad'
                return self.grad.norm()
        elif norm_kind in ('spectral', 'spectral_exact'):
            # initialize power iterations on rawg (momentum/grad)
            # ref: https://github.com/jxbz/modula/blob/e274a352551ec4c6055b7fc0086db7a516863578/modula/atom.py#L32
            #      https://github.com/pytorch/pytorch/blob/d7e0e1dbc453bac099f747dfb65ad75767c3e1d7/torch/nn/utils/spectral_norm.py#L96
            if tensor_kind == 'g':
                assert self.zeropower_backend not in ('svd', 'sign'), "spectral norm of g not supported for svd or sign"
                assert self.momentum_kind not in {'post_ns', 'post_ns_nesterov'}, "spectral norm of g not supported for post-ns which breaks g=rawg0"
                if norm_kind == 'spectral':
                    # g should only have binary singular values, just assume 1! if it is 0, then scaling it with 1 is still 0
                    return 1
                elif norm_kind == 'spectral_exact':
                    # g should only have binary singular values... having at least 1 non-zero means that frobenius norm is at least 1 = \sqrt{ \sum_i s_i^2 }
                    return torch.ge(self.g.norm(), 0.9, out=self.g.new_empty(()))

            tensor = dict(rawg=self.rawg, grad=self.grad)[tensor_kind]
            if f'{tensor_kind}_u' not in self.state:
                self.state[f'{tensor_kind}_u'] = F.normalize(torch.randn_like(tensor[0]), dim=0, eps=self.eps)
                self.state[f'{tensor_kind}_v'] = torch.empty_like(tensor[:, 0])
                niter = 5
            else:
                niter = 1
            u = self.state[f'{tensor_kind}_u']
            v = self.state[f'{tensor_kind}_v']
            for _ in range(niter):
                torch.mv(tensor, u, out=v)
                F.normalize(v, dim=0, eps=self.eps, out=v)
                torch.mv(tensor.T, v, out=u)
                F.normalize(u, dim=0, eps=self.eps, out=u)
            return torch.dot(v, torch.mv(tensor, u))
        elif norm_kind in {'nuclear', 'nuclear_exact'}:
            if tensor_kind == 'rawg':
                if self.g.shape[0] > self.g.shape[1]:
                    # (U V^T)^T U S V^T = V S V^T
                    # trace(V S V^T) = trace(S)
                    return (self.rawg0.T @ self.rawg).trace()
                else:
                    # U S V^T (U V^T)^T = U S U^T
                    # trace(U S U^T) = trace(S)
                    return (self.rawg @ self.rawg0.T).trace()
            elif tensor_kind == 'g':
                assert self.zeropower_backend not in ('svd', 'sign'), "nuclear (est) norm of g not supported for svd or sign"
                assert self.momentum_kind not in {'post_ns', 'post_ns_nesterov'}, "nuclear (est) norm of g not supported for post-ns which breaks g=rawg0"
                return self(tensor_kind, norm_kind.replace('nuclear', 'fro'), dual=dual) ** 2
            else:
                assert False, f"{norm_kind} not implemented for tensor_kind {tensor_kind}"
        else:
            assert False, f"unknown norm kind {norm_kind}"


@torch.compile
def _right_preconditioner_from_zerothpower(g, g0, sqrt_dim: float, dtype=torch.float32, eps=1e-7):
    # return V S-1 V.T
    vsv = (g0.T @ g).to(dtype)
    vsv = vsv / vsv.norm()
    vsv.diagonal(dim1=-2, dim2=-1).add_(eps)
    inv = torch.linalg.pinv(vsv, hermitian=True).to(g.dtype)
    # L, info = torch.linalg.cholesky_ex(vsv)
    # if info.item() != 0:
    #     raise RuntimeError(f"cholesky_ex failed with info {info}")
    # inv = torch.cholesky_inverse(L).to(g.dtype)
    return inv / inv.norm() * sqrt_dim  # unit fro -> unit mean s^2

def right_preconditioner_from_zerothpower_with_retry(g, g0, eps=1e-7):
    dtype = torch.float32
    sqrt_dim = g.size(1)**0.5
    while True:
        try:
            return _right_preconditioner_from_zerothpower(g, g0, sqrt_dim, dtype=dtype, eps=eps)
        except RuntimeError:
            if dtype == torch.float64:
                return None
            dtype = torch.float64

@torch.compile
def _left_preconditioner_from_zerothpower(g, g0, sqrt_dim: float, dtype=torch.float32, eps=1e-7):
    # return U S-1 U.T
    usu = (g @ g0.T).to(dtype)
    usu = usu / usu.norm()
    usu.diagonal(dim1=-2, dim2=-1).add_(eps)
    inv = torch.linalg.pinv(usu, hermitian=True).to(g.dtype)
    # L, info = torch.linalg.cholesky_ex(usu)
    # if info.item() != 0:
    #     raise RuntimeError(f"cholesky_ex failed with info {info}")
    # inv = torch.cholesky_inverse(L).to(g.dtype)
    return inv / inv.norm() * sqrt_dim  # unit fro -> unit mean s^2

def left_preconditioner_from_zerothpower_with_retry(g, g0, eps=1e-7):
    dtype = torch.float32
    sqrt_dim = g.size(0)**0.5
    while True:
        try:
            return _left_preconditioner_from_zerothpower(g, g0, sqrt_dim, dtype=dtype, eps=eps)
        except RuntimeError:
            if dtype == torch.float64:
                return None
            dtype = torch.float64


class Muon(torch.optim.Optimizer):
    r"""
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, beta2=0.999, momentum_kind='pre_ns_nesterov', backend='newtonschulz5',
                 backend_steps=5, norm_kind='rms', target_norm='unit', eps=1e-7,
                 compute_precondition_freq=20,
                 precondition_backend=None, precondition_backend_steps=10,
                 precondition_kind=None,
                 precondition_beta2=0.999):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, momentum_kind=momentum_kind, backend=backend, backend_steps=backend_steps,
                        norm_kind=norm_kind, target_norm=target_norm, eps=eps, compute_precondition_freq=compute_precondition_freq,
                        precondition_backend=precondition_backend, precondition_backend_steps=precondition_backend_steps,
                        precondition_kind=precondition_kind, precondition_beta2=precondition_beta2)
        assert momentum_kind in {'pre_ns', 'pre_ns_nesterov', 'post_ns', 'post_ns_nesterov', 'post_norm_scale', 'post_norm_scale_nesterov', None}
        assert precondition_kind in {'left', 'left_lstsq' 'right', 'min_dim', None}
        super().__init__(params, defaults)

    def _apply_momentum(self, state, x: torch.Tensor, momentum, *, is_nesterov, prefix: str):
        # m = beta1 * m + g
        # g = g + m * beta1
        # ----
        # equiv:
        # g = g + beta1**2 * m + beta1 * g
        #   = (1+beta1) g + beta1**2 m
        if f'{prefix}_momentum_buffer' not in state:
            state[f'{prefix}_momentum_buffer'] = torch.zeros_like(x)
        buf = state[f'{prefix}_momentum_buffer']
        if is_nesterov:
            buf.mul_(momentum).add_(x)
            return x.add(buf, alpha=momentum)
        else:
            torch.lerp(buf, x, 1 - momentum, out=buf)
            return buf / (1 - momentum ** state['stept'])


    def step(self):
        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr']
            momentum = group['momentum']

            # renormalize update
            norms = {}

            for p in group['params']:
                # grad      := p.grad
                # rawg      := maybe-pre-ns-momentum(grad)
                # cond_rawg := maybe-preconditioner(rawg)
                # rawg0     := zeropower(cond_rawg)
                # g         := maybe-post-ns-momentum(rawg0)

                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if 'stept' not in state:
                    state['stept'] = 0
                state['stept'] += 1
                stept = state['stept']

                # pre-ns momentum
                rawg = grad
                if group['momentum_kind'] in {'pre_ns', 'pre_ns_nesterov'}:
                    rawg = self._apply_momentum(state, rawg, momentum, is_nesterov=group['momentum_kind'] == 'pre_ns_nesterov', prefix='pre_ns')

                # apply preconditioner
                cond_rawg = rawg
                cond_rawgnorm_fro = rawgnorm_fro = rawg.norm()
                do_update_preconditioner = False
                precondition_kind = group['precondition_kind']
                if precondition_kind is not None:
                    if precondition_kind == 'min_dim':
                        precondition_kind = 'left' if p.shape[0] < p.shape[1] else 'right'
                    else:
                        precondition_kind = precondition_kind
                    if (preconditioner:= state.get('preconditioner', None)) is not None:
                        if group['precondition_beta2'] != 1:
                            # regress to idt
                            preconditioner.mul_(group['precondition_beta2'])
                            preconditioner.diagonal(dim1=-2, dim2=-1).add_(1 - group['precondition_beta2'])
                        if precondition_kind == 'left':
                            cond_rawg = preconditioner @ cond_rawg
                        elif precondition_kind == 'right':
                            cond_rawg = cond_rawg @ preconditioner
                        else:
                            assert False, f"unknown precondition_kind {group['precondition_kind']}"
                        cond_rawgnorm_fro = cond_rawg.norm()

                    do_update_preconditioner = stept > 0 and stept % group['compute_precondition_freq'] == 0

                backend = group['backend']
                backend_steps = group['backend_steps']
                if do_update_preconditioner:
                    if group['precondition_backend'] is not None:
                        backend = group['precondition_backend']
                    if group['precondition_backend_steps'] is not None:
                        backend_steps = group['precondition_backend_steps']

                # ns iter
                rawg0 = zeropower_backends[backend](cond_rawg, steps=backend_steps, dtype=cond_rawg.dtype, eps=eps, G_fro=cond_rawgnorm_fro)

                # update preconditioner
                if do_update_preconditioner:
                    assert group['backend'] not in {'sgd', 'sign'}, "preconditioner not supported for sgd or sign"
                    # here we don something "hacky" to pick eps
                    # let's assume that rawg0 has singular values in [0.95, 1.05]
                    # rawg/rawgnorm_fro has singular values in [0, 1]
                    # now the preconditioner is generally (rawg0.T @ rawg + eps I)^{-1}
                    # note that scaling preconditioner doesn't matter for the 0th power
                    # so we can also do (rawg0.T @ rawg / C1 + eps I)^{-1} / C2
                    if precondition_kind == 'left':
                        state['preconditioner'] = left_preconditioner_from_zerothpower_with_retry(rawg, rawg0, eps=1e-3)
                    elif precondition_kind == 'right':
                        state['preconditioner'] = right_preconditioner_from_zerothpower_with_retry(rawg, rawg0, eps=1e-3)

                # post-ns momentum
                g = rawg0
                if group['momentum_kind'] in {'post_ns', 'post_ns_nesterov'}:
                    g = self._apply_momentum(state, g, momentum, is_nesterov=group['momentum_kind'] == 'post_ns_nesterov', prefix='post_ns')
                state['last_update'] = dict(grad=p.grad, rawg=rawg, cond_rawg=cond_rawg, rawg0=rawg0, g=g, rawgnorm_fro=rawgnorm_fro)
                norms[p] = NormInterface(
                    state,
                    zeropower_backend=group['backend'],
                    momentum_kind=group['momentum_kind'],
                    eps=eps,
                    **state['last_update'],
                )

            norm_kind = group['norm_kind']
            target_norm = group['target_norm']

            if target_norm is None:
                # no rescaling
                for p, norm_interface in norms.items():
                    self.state[p]['last_update']['target_norm'] = None

            elif target_norm == 'unit':
                # target_norm = 1
                for p, norm_interface in norms.items():
                    self.state[p]['last_update']['target_norm'] = 1

            else:
                use_dual = target_norm.endswith('_dual')
                if use_dual:
                    target_norm = target_norm[:-5]

                if target_norm == 'ema_grad_norm':
                    # target_norm = ema(||grad||)
                    for p, norm_interface in norms.items():
                        state = self.state[p]
                        if 'ema_grad_norm' not in state:
                            state['ema_grad_norm'] = p.grad.new_zeros(())
                        torch.lerp(state['ema_grad_norm'], norm_interface('grad', norm_kind, dual=use_dual), 1 - group['beta2'], out=state['ema_grad_norm'])
                        self.state[p]['last_update']['target_norm'] = state['ema_grad_norm'] / (1 - group['beta2']**stept)

                elif target_norm == 'ema_rawg_norm':
                    # target_norm = ema(||rawg||)
                    for p, norm_interface in norms.items():
                        state = self.state[p]
                        if 'ema_rawg_norm' not in state:
                            state['ema_rawg_norm'] = p.grad.new_zeros(())
                        torch.lerp(state['ema_rawg_norm'], norm_interface('rawg', norm_kind, dual=use_dual), 1 - group['beta2'], out=state['ema_rawg_norm'])
                        self.state[p]['last_update']['target_norm'] = state['ema_rawg_norm'] / (1 - group['beta2']**stept)

                elif target_norm == 'ema_grad_norm2_sqrt':
                    # target_norm = ema(||grad||^2)^0.5
                    for p, norm_interface in norms.items():
                        state = self.state[p]
                        if 'ema_grad_norm2' not in state:
                            state['ema_grad_norm2'] = p.grad.new_zeros(())
                        torch.lerp(state['ema_grad_norm2'], norm_interface('grad', norm_kind, dual=use_dual)**2, 1 - group['beta2'], out=state['ema_grad_norm2'])
                        self.state[p]['last_update']['target_norm'] = (state['ema_grad_norm2'] / (1 - group['beta2']**stept))**0.5

                elif target_norm == 'ema_rawg_norm2_sqrt':
                    # target_norm = ema(||rawg||^2)^0.5
                    for p, norm_interface in norms.items():
                        state = self.state[p]
                        if 'ema_rawg_norm' not in state:
                            state['ema_rawg_norm2'] = p.grad.new_zeros(())
                        torch.lerp(state['ema_rawg_norm2'], norm_interface('rawg', norm_kind, dual=use_dual)**2, 1 - group['beta2'], out=state['ema_rawg_norm2'])
                        self.state[p]['last_update']['target_norm'] = (state['ema_rawg_norm2'] / (1 - group['beta2']**stept))**0.5

                elif target_norm == 'rawg':
                    # target_norm = ||rawg||
                    for p, norm_interface in norms.items():
                        self.state[p]['last_update']['target_norm'] = norm_interface('rawg', norm_kind, dual=use_dual)

                else:
                    if target_norm == 'globalavg_rawg':
                        target_norm = (
                            sum(norm_interface('rawg', norm_kind, dual=use_dual) for _, norm_interface in norms.items())
                            /
                            len(norms)
                        )
                    elif target_norm == 'globalmax_rawg':
                        target_norm = max(norm_interface('rawg', norm_kind, dual=use_dual) for _, norm_interface in norms.items())
                    else:
                        assert False, f"unknown target_norm {group['target_norm']}"

                    for p, norm_interface in norms.items():
                        self.state[p]['last_update']['target_norm'] = target_norm

            for p, norm_interface in norms.items():
                state = self.state[p]
                last_update = state['last_update']
                target_norm = last_update['target_norm']
                if target_norm is None:
                    scale = 1
                else:
                    scale = target_norm / norm_interface('g', norm_kind, dual=False)  # see anthology proposition 1. unit norm, scale to target_norm that could be the dual||g||^dagger
                update = norm_interface.g * scale

                if group['momentum_kind'] in {'post_norm_scale', 'post_norm_scale_nesterov'}:
                    # e.g., use (momentum_kind='post_norm_scale_*', target_norm='rawg_dual') to implement
                    #      g := ema(||rawg||^\dagger * rawg^0 / ||rawg^0||)
                    update = self._apply_momentum(state, update, momentum, is_nesterov=group['momentum_kind'] == 'post_norm_scale_nesterov', prefix='post_norm_scale')

                state['last_update']['update'] = update
                p.data.add_(update, alpha=-lr)



Result = namedtuple('Result', ['steps', 'train_accs', 'eval_accs', 'model_ws', 'state_dict', 'losses_all_steps', 'time_all_steps'])

class CallBackProtocol(Protocol):
    def __call__(self, step: int, model: nn.Module, optimizers: list[torch.optim.Optimizer], data: torch.Tensor, target: torch.Tensor, loss: torch.Tensor) -> None: ...


def train_mnist(model, opt, nsteps=3000, log_nsteps=100,
                dtype=torch.bfloat16,
                eval_on_log=True, one_batch_overfit=False,
                batch_size=512, lr: float = 1e-3, loss_scale=1,
                w_save_key: Optional[str] = None,
                post_step_callback: Optional[CallBackProtocol] = None):
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, dtype)
    print(f'training on {device}, {dtype}')

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False)

    test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    # Assuming model is already defined and named "model"
    model = model.to(device)
    optimizers = []
    if callable(opt):
        opt_cls = opt([torch.empty(1, 1, requires_grad=True)]).__class__
        if opt_cls is Muon:
            opt_1d = optim.Adam
        else:
            opt_1d = opt
        optimizers.extend([
            opt([p for p in model.parameters() if len(p.data.shape) == 2], lr=lr),
            opt_1d([p for p in model.parameters() if len(p.data.shape) != 2], lr=lr),
        ])
    else:
        raise ValueError(f"Unknown optimizer: {opt}")

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        for data, target in loader:
            data, target = data.to(device, dtype), target.to(device)
            output = model(data.flatten(1))
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        model.train()
        return correct / total

    # Training loop
    losses_all_steps = []
    time_all_steps = []
    steps = []
    model_ws = []
    train_accs = []
    eval_accs = []

    def record(step):
        if eval_on_log:
            train_acc = evaluate(model, train_eval_loader)
            eval_acc = evaluate(model, test_loader)
            eval_accs.append(eval_acc)
            train_accs.append(train_acc)
            print(f"Step {step}/{nsteps}, Train Accuracy: {train_acc:.2%}, Test Accuracy: {eval_acc:.2%}")
            print('-' * 100)
        steps.append(step)
        if w_save_key is not None:
            model_ws.append(
                model.state_dict()[w_save_key].to('cpu', torch.float32).data
            )

    record(0)

    # if post_step_callback is not None:
    #     post_step_callback(0, model, optimizers, None, None, None)

    import itertools

    if one_batch_overfit:
        data, target = next(iter(train_loader))
        data, target = data.to(device, dtype), target.to(device)  # save some time per iteration
        train_loader = itertools.repeat((data, target))

    inf_train_loader = itertools.cycle(train_loader)

    from tqdm.auto import tqdm

    model.train()
    for step, (data, target) in tqdm(enumerate(inf_train_loader, start=1), total=nsteps, disable=True):
        t0 = time.time()
        data, target = data.to(device, dtype), target.to(device)

        model.zero_grad()
        output = model(data.flatten(1))
        loss = F.cross_entropy(output, target)
        losses_all_steps.append(loss.item())
        (loss * loss_scale).backward()

        for optimizer in optimizers:
            optimizer.step()

        time_all_steps.append(time.time() - t0)

        if post_step_callback is not None:
            post_step_callback(step, model, optimizers, data, target, loss)

        if step % log_nsteps == 0:
            print(f"Step {step}/{nsteps}, Loss: {loss.item():.4f}")
            record(step)

        if step == nsteps:
            break

    print("Training completed!")

    return Result(steps, train_accs, eval_accs, model_ws, model.state_dict(), losses_all_steps, time_all_steps)


def make_model():
    model = nn.Sequential(
        nn.Linear(28*28, 28*28),
        nn.ReLU(),
        nn.Linear(28*28, 28*28),
        nn.ReLU(),
        nn.Linear(28*28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(torch.bfloat16)

    model = torch.compile(model)
    return model


def orth_init(model):
    import torch.nn as nn
    @torch.no_grad()
    def init(m):
        if isinstance(m, nn.Linear):
            fan_out = m.weight.size(0)
            fan_in = m.weight.size(1)
            dtype = m.weight.dtype
            m.to(torch.float32)
            nn.init.orthogonal_(m.weight)
            m.weight.mul_(
                (fan_out / fan_in) ** 0.5
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            m.to(dtype)
    model.apply(init)
    return model


# torch.save([orth_init(model).state_dict()], 'orth_init_weights.pth')


# name -> (optim_cls, desc)
OPTIM_MAP: Mapping[str, Tuple[Union[str, Callable], List[str]]] = dict(
    shampoo_precond1=                                               (functools.partial(DistributedShampoo, betas=(0.9, 0.999), epsilon=1e-12, weight_decay=0,
                                                                                       max_preconditioner_dim=2048, precondition_frequency=1,
                                                                                       start_preconditioning_step=-1, use_decoupled_weight_decay=True,
                                                                                       grafting_config=AdamGraftingConfig(
                                                                                            beta2=0.9,
                                                                                            epsilon=1e-12,
                                                                                        )),                                                                                           r'''shampoo (precond freq=1)'''),
    shampoo_precond10=                                             (functools.partial(DistributedShampoo, betas=(0.9, 0.999), epsilon=1e-12, weight_decay=0,
                                                                                       max_preconditioner_dim=2048, precondition_frequency=10,
                                                                                       start_preconditioning_step=10, use_decoupled_weight_decay=True,
                                                                                       grafting_config=AdamGraftingConfig(
                                                                                            beta2=0.9,
                                                                                            epsilon=1e-12,
                                                                                        )),                                                                                           r'''shampoo (precond freq=10)'''),
    shampoo_precond25=                                             (functools.partial(DistributedShampoo, betas=(0.9, 0.999), epsilon=1e-12, weight_decay=0,
                                                                                       max_preconditioner_dim=2048, precondition_frequency=25,
                                                                                       start_preconditioning_step=25, use_decoupled_weight_decay=True,
                                                                                       grafting_config=AdamGraftingConfig(
                                                                                            beta2=0.9,
                                                                                            epsilon=1e-12,
                                                                                        )),                                                                                           r'''shampoo (precond freq=25)'''),
    shampoo_precond50=                                             (functools.partial(DistributedShampoo, betas=(0.9, 0.999), epsilon=1e-12, weight_decay=0,
                                                                                       max_preconditioner_dim=2048, precondition_frequency=50,
                                                                                       start_preconditioning_step=50, use_decoupled_weight_decay=True,
                                                                                       grafting_config=AdamGraftingConfig(
                                                                                            beta2=0.9,
                                                                                            epsilon=1e-12,
                                                                                        )),                                                                                           r'''shampoo (precond freq=50)'''),
    shampoo_precond100=                                             (functools.partial(DistributedShampoo, betas=(0.9, 0.999), epsilon=1e-12, weight_decay=0,
                                                                                       max_preconditioner_dim=2048, precondition_frequency=100,
                                                                                       start_preconditioning_step=100, use_decoupled_weight_decay=True,
                                                                                       grafting_config=AdamGraftingConfig(
                                                                                            beta2=0.9,
                                                                                            epsilon=1e-12,
                                                                                        )),                                                                                           r'''shampoo (precond freq=100)'''),

    adam=                                                           (functools.partial(optim.Adam, betas=(0.9, 0.999)),                                                               r'''adam'''),
    adam_b09=                                                       (functools.partial(optim.Adam, betas=(0.9, 0.999)),                                                               r'''adam ($\beta_1$=0.9, default)'''),
    adam_b095=                                                      (functools.partial(optim.Adam, betas=(0.95, 0.999)),                                                              r'''adam ($\beta_1$=0.95)'''),
    adam_b0995=                                                     (functools.partial(optim.Adam, betas=(0.995, 0.999)),                                                             r'''adam ($\beta_1$=0.995)'''),
    muon=                                                           (functools.partial(Muon, backend='newtonschulz5'),                                                                r'''muon'''),

    muon_1step=                                                     (functools.partial(Muon, backend='newtonschulz5', backend_steps=1),                                               r'''muon w/ 1-step NS iter'''),
    muon_2step=                                                     (functools.partial(Muon, backend='newtonschulz5', backend_steps=2),                                               r'''muon w/ 2-step NS iter'''),
    muon_3step=                                                     (functools.partial(Muon, backend='newtonschulz5', backend_steps=3),                                               r'''muon w/ 3-step NS iter'''),
    muon_5step=                                                     (functools.partial(Muon, backend='newtonschulz5', backend_steps=5),                                               r'''muon w/ 5-step NS iter (default)'''),
    muon_1step_precond50=                                           (functools.partial(Muon, backend='newtonschulz5', backend_steps=1,
                                                                                       precondition_backend='newtonschulz5_sched10', precondition_backend_steps=10,
                                                                                       precondition_kind='min_dim', compute_precondition_freq=50, precondition_beta2=0.7),            r'''muon w/ 1-step preconditioned NS iter
                                                                                                                                                                                          (compute preconditioners every 50 steps)'''),
    muon_2step_precond50=                                            (functools.partial(Muon, backend='newtonschulz5', backend_steps=2,
                                                                                        precondition_backend='newtonschulz5_sched10', precondition_backend_steps=10,
                                                                                        precondition_kind='min_dim', compute_precondition_freq=50, precondition_beta2=0.7),           r'''muon w/ 2-step preconditioned NS iter
                                                                                                                                                                                          (compute preconditioners every 50 steps)'''),
    muon_3step_precond50=                                            (functools.partial(Muon, backend='newtonschulz5', backend_steps=3,
                                                                                        precondition_backend='newtonschulz5_sched10', precondition_backend_steps=10,
                                                                                        precondition_kind='min_dim', compute_precondition_freq=50, precondition_beta2=0.7),           r'''muon w/ 3-step preconditioned NS iter
                                                                                                                                                                                          (compute preconditioners every 50 steps)'''),

    muon_pre_ns=                                                    (functools.partial(Muon, momentum_kind='pre_ns'),                                                                 r'''muon w/ pre-ns ema
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                        '''),

    muon_post_ns=                                                   (functools.partial(Muon, momentum_kind='post_ns', norm_kind='rms_exact'),                                         r'''muon w/ post-ns ema
                                                                                                                                                                                          $\text{grad}^0_i := \text{NS}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{ema}(\text{grad}^0_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                        '''),

    muon_pre_ns_nesterov=                                           (functools.partial(Muon, momentum_kind='pre_ns_nesterov'),                                                        r'''muon w/ pre-ns nesterov-type update (default)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                        '''),

    muon_post_ns_nesterov=                                          (functools.partial(Muon, momentum_kind='post_ns_nesterov', norm_kind='rms_exact'),                                r'''muon w/ post-ns nesterov-type update
                                                                                                                                                                                          $\text{grad}^0_i := \text{NS}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{nesterov}(\text{grad}^0_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                        '''),


    muon_sgd=                                                       (functools.partial(Muon, backend='sgd', target_norm=None),                                                        r'''SGD (i.e., muon w/o orthogonalization)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{momentum}_i$
                                                                                                                                                                                        '''),

    muon_sign=                                                      (functools.partial(Muon, backend='sign', target_norm=None),                                                       r'''sign-SGD (i.e., muon w/ sign instead of orthogonalization)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{sign}(\text{momentum}_i)$
                                                                                                                                                                                        '''),

    muon_proper=                                                    (functools.partial(Muon, backend='newtonschulz5_proper'),                                                         r'''muon w/ naive simple cubic NS iter (still 5 steps)'''),
    muon_sched5=                                                    (functools.partial(Muon, backend='newtonschulz5_sched5', backend_steps=5),                                        r'''muon w/ scheduled 5-step NS iter
                                                                                                                                                                                          (default$\rightarrow$naive)'''),
    muon_sched8=                                                    (functools.partial(Muon, backend='newtonschulz5_sched8', backend_steps=8),                                        r'''muon w/ scheduled 8-step NS iter
                                                                                                                                                                                          (default$\rightarrow$naive)'''),
    muon_sched10=                                                   (functools.partial(Muon, backend='newtonschulz5_sched10', backend_steps=10),                                      r'''muon w/ scheduled 10-step NS iter
                                                                                                                                                                                          (default$\rightarrow$naive)'''),
    muon_sched14=                                                   (functools.partial(Muon, backend='newtonschulz5_sched14', backend_steps=14),                                      r'''muon w/ scheduled 14-step NS iter
                                                                                                                                                                                          (default$\rightarrow$naive)'''),

    muon_momentum099=                                               (functools.partial(Muon, backend='newtonschulz5', momentum=0.99),                                                 r'''muon w/ 0.99 nesterov momentum
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)_{\text{momentum}=0.99}$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                          '''),
    muon_momentum095=                                               (functools.partial(Muon, backend='newtonschulz5', momentum=0.95),                                                 r'''muon w/ 0.95 nesterov momentum (default)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)_{\text{momentum}=0.95}$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                          '''),
    muon_momentum09=                                                (functools.partial(Muon, backend='newtonschulz5', momentum=0.9),                                                  r'''muon w/ 0.9 nesterov momentum
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)_{\text{momentum}=0.9}$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                          '''),
    muon_momentum085=                                               (functools.partial(Muon, backend='newtonschulz5', momentum=0.85),                                                 r'''muon w/ 0.85 nesterov momentum
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)_{\text{momentum}=0.85}$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                          '''),
    muon_momentum08=                                                (functools.partial(Muon, backend='newtonschulz5', momentum=0.8),                                                  r'''muon w/ 0.8 nesterov momentum
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)_{\text{momentum}=0.8}$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                          '''),
    muon_no_momentum=                                               (functools.partial(Muon, backend='newtonschulz5', momentum_kind=None),                                            r'''muon w/o momentum
                                                                                                                                                                                          $\text{grad}^0_i := \text{NS}(\text{grad}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{grad}^0_i}{||\text{grad}^0_i||_\text{rms}}$
                                                                                                                                                                                          '''),

    muon_norm_rms_target_unit=                                      (functools.partial(Muon, norm_kind='rms', target_norm='unit'),                                                    r'''muon w/ unit norm update (approx rms, default)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_rms_exact_target_unit=                                (functools.partial(Muon, norm_kind='rms_exact', target_norm='unit'),                                              r'''muon w/ unit norm update (rms)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_fro_target_unit=                                      (functools.partial(Muon, norm_kind='fro', target_norm='unit'),                                                    r'''muon w/ unit norm update (approx frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{F}}$
                                                                                                                                                                                       '''),
    muon_norm_fro_exact_target_unit=                                (functools.partial(Muon, norm_kind='fro_exact', target_norm='unit'),                                              r'''muon w/ unit norm update (frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{F}}$
                                                                                                                                                                                       '''),
    muon_norm_spec_target_unit=                                     (functools.partial(Muon, norm_kind='spectral', target_norm='unit'),                                               r'''muon w/ unit norm update (spectral)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                       '''),
    muon_norm_jb_target_unit=                                       (functools.partial(Muon, norm_kind='jbnorm', target_norm='unit'),                                                 r'''muon w/ unit norm update ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                       '''),
    muon_pre_ns_norm_jb_target_unit=                                (functools.partial(Muon, norm_kind='jbnorm', target_norm='unit', momentum_kind='pre_ns'),                         r'''muon w/ pre-ns ema & unit norm update ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                       '''),


    muon_norm_rms_target_momentum=                                  (functools.partial(Muon, norm_kind='rms', target_norm='rawg'),                                                  r'''muon w/ norm-match (approx rms)
                                                                                                                                                                                        $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                        $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                        $\Delta W_i := ||\text{momentum}_i||_\text{rms} \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_rms_exact_target_momentum=                            (functools.partial(Muon, norm_kind='rms_exact', target_norm='rawg'),                                          r'''muon w/ norm-match (rms)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_\text{rms} \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_fro_target_momentum=                                  (functools.partial(Muon, norm_kind='fro', target_norm='rawg'),                                                r'''muon w/ norm-match (approx frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_F \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                       '''),
    muon_norm_fro_exact_target_momentum=                            (functools.partial(Muon, norm_kind='fro_exact', target_norm='rawg'),                                          r'''muon w/ norm-match (frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_F \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                       '''),
    muon_norm_spec_target_momentum=                                 (functools.partial(Muon, norm_kind='spectral', target_norm='rawg'),                                           r'''muon w/ norm-match (spectral)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_* \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                       '''),
    muon_norm_jb_target_momentum=                                   (functools.partial(Muon, norm_kind='jbnorm', target_norm='rawg'),                                             r'''muon w/ norm-match ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}} \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                       '''),

    muon_norm_rms_target_momentum_dual=                                  (functools.partial(Muon, norm_kind='rms', target_norm='rawg_dual'),                                                r'''muon w/ dual-norm-match (approx rms)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_\text{rms}^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_rms_exact_target_momentum_dual=                                  (functools.partial(Muon, norm_kind='rms_exact', target_norm='rawg_dual'),                                                r'''muon w/ dual-norm-match (rms)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_\text{rms}^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_fro_target_momentum_dual=                                  (functools.partial(Muon, norm_kind='fro', target_norm='rawg_dual'),                                                r'''muon w/ dual-norm-match (approx frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_F^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                       '''),
    muon_norm_fro_exact_target_momentum_dual=                            (functools.partial(Muon, norm_kind='fro_exact', target_norm='rawg_dual'),                                          r'''muon w/ dual-norm-match (frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_F^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                       '''),
    muon_norm_spec_target_momentum_dual=                                 (functools.partial(Muon, norm_kind='spectral', target_norm='rawg_dual'),                                           r'''muon w/ dual-norm-match (spectral)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_*^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                       '''),
    muon_norm_jb_target_momentum_dual=                                   (functools.partial(Muon, norm_kind='jbnorm', target_norm='rawg_dual'),                                         r'''muon w/ dual-norm-match ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                       '''),
    # muon_2step_norm_jb_target_momentum_dual=                        (functools.partial(Muon, norm_kind='jbnorm', target_norm='rawg_dual', backend_steps=2),                       [r'muon w/ 2-step NS iter & norm-match ($\text{rms}\rightarrow\text{rms}$)',
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    # muon_2step_precond50_norm_jb_target_momentum_dual=              (functools.partial(Muon, norm_kind='jbnorm', target_norm='rawg_dual', backend_steps=2,
    #                                                                                    precondition_backend='newtonschulz5_sched10', precondition_backend_steps=10,
    #                                                                                    precondition_kind='min_dim', compute_precondition_freq=50, precondition_beta2=0.7),           [r'muon w/ 2-step preconditioned NS iter & norm-match ($\text{rms}\rightarrow\text{rms}$)',
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    muon_pre_ns_norm_jb_target_momentum_dual=                       (functools.partial(Muon, norm_kind='jbnorm', target_norm='rawg_dual', momentum_kind='pre_ns'),                    r'''muon w/ pre-ns ema & dual-norm-match ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                       '''),
    muon_post_norm_scale_nesterov_norm_jb_target_grad_dual=         (functools.partial(Muon, norm_kind='jbnorm', target_norm='rawg_dual', momentum_kind='post_norm_scale_nesterov'),  r'''muon w/ norm-weighted nesterov update ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{grad}^0_i := \text{NS}(\text{grad}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{nesterov}(||\text{grad}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger \frac{\text{grad}^0_i}{||\text{grad}^0_i||_{\text{rms}\rightarrow\text{rms}}})$
                                                                                                                                                                                       '''),


    muon_norm_rms_target_ema_grad_norm2_sqrt=                       (functools.partial(Muon, norm_kind='rms', target_norm='ema_grad_norm2_sqrt'),                                     r'''muon w/ ema-norm-match (approx rms)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{ema}(||\text{grad}_i||_\text{rms}) \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_rms_exact_target_ema_grad_norm2_sqrt=                 (functools.partial(Muon, norm_kind='rms_exact', target_norm='ema_grad_norm2_sqrt'),                                r'''muon w/ ema-norm-match (rms)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{ema}(||\text{grad}_i||_\text{rms}) \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                       '''),
    muon_norm_fro_target_ema_grad_norm2_sqrt=                       (functools.partial(Muon, norm_kind='fro', target_norm='ema_grad_norm2_sqrt'),                                     r'''muon w/ ema-norm-match (approx frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{ema}(||\text{momentum}_i||_F) \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                       '''),
    muon_norm_fro_exact_target_ema_grad_norm2_sqrt=                 (functools.partial(Muon, norm_kind='fro_exact', target_norm='ema_grad_norm2_sqrt'),                                r'''muon w/ ema-norm-match (frobenius)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{ema}(||\text{momentum}_i||_F) \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                       '''),
    muon_norm_spec_target_ema_grad_norm2_sqrt=                      (functools.partial(Muon, norm_kind='spectral', target_norm='ema_grad_norm2_sqrt'),                                r'''muon w/ ema-norm-match (spectral)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{ema}(||\text{momentum}_i||_*) \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                       '''),
    muon_norm_jb_target_ema_grad_norm2_sqrt=                        (functools.partial(Muon, norm_kind='jbnorm', target_norm='ema_grad_norm2_sqrt'),                                  r'''muon w/ ema-norm-match ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{grad}^0_i := \text{NS}(\text{grad}_i)$
                                                                                                                                                                                          $\Delta W_i := \text{ema}(||\text{grad}_i||_{\text{rms}\rightarrow\text{rms}}) \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                       '''),
    # muon_norm_jb_target_ema_grad_norm2_sqrt_dual=                   (functools.partial(Muon, norm_kind='jbnorm', target_norm='ema_grad_norm2_sqrt_dual'),                             [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $\text{ema}((||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger)^2)^{1/2}$)']),
    # muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual=            (functools.partial(Muon, norm_kind='jbnorm', target_norm='ema_grad_norm2_sqrt_dual', momentum_kind='pre_ns'),     [r'muon w/ pre-ns ema & norm-match ($\text{rms}\rightarrow\text{rms}$)',
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $\text{ema}((||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger)^2)^{1/2}$)']),


    muon_norm_rms_target_glbavgmomentum=                            (functools.partial(Muon, norm_kind='rms', target_norm='globalavg_rawg'),                                          r'''muon w/ modula-like norm-match (approx rms, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_\text{rms}$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),

    muon_norm_rms_exact_target_glbavgmomentum=                      (functools.partial(Muon, norm_kind='rms_exact', target_norm='globalavg_rawg'),                                r'''muon w/ modula-like norm-match (rms, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_\text{rms}$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),
    muon_norm_fro_target_glbavgmomentum=                            (functools.partial(Muon, norm_kind='fro', target_norm='globalavg_rawg'),                                      r'''muon w/ modula-like norm-match (approx frobenius, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_F$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_fro_exact_target_glbavgmomentum=                      (functools.partial(Muon, norm_kind='fro_exact', target_norm='globalavg_rawg'),                                r'''muon w/ modula-like norm-match (frobenius, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_F$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_spec_target_glbavgmomentum=                           (functools.partial(Muon, norm_kind='spectral', target_norm='globalavg_rawg'),                                 r'''muon w/ modula-like norm-match (spectral, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_*$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                    '''),
    muon_norm_jb_target_glbavgmomentum=                             (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_rawg'),                                   r'''muon w/ modula-like norm-match ($\text{rms}\rightarrow\text{rms}$, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                    '''),

    muon_norm_rms_target_glbavgmomentum_dual=                        (functools.partial(Muon, norm_kind='rms', target_norm='globalavg_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match (approx rms, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_\text{rms}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),
    muon_norm_rms_exact_target_glbavgmomentum_dual=                 (functools.partial(Muon, norm_kind='rms_exact', target_norm='globalavg_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match (rms, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_\text{rms}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),
    muon_norm_fro_target_glbavgmomentum_dual=                        (functools.partial(Muon, norm_kind='fro', target_norm='globalavg_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match (approx frobenius, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_F^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_fro_exact_target_glbavgmomentum_dual=                 (functools.partial(Muon, norm_kind='fro_exact', target_norm='globalavg_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match (frobenius, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_F^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_spec_target_glbavgmomentum_dual=                        (functools.partial(Muon, norm_kind='spectral', target_norm='globalavg_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match (spectral, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_*^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                    '''),
    muon_norm_jb_target_glbavgmomentum_dual=                        (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match ($\text{rms}\rightarrow\text{rms}$, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                    '''),
    # muon_2step_norm_jb_target_glbavgmomentum_dual=                   (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_rawg_dual', backend_steps=2),            r'''muon w/ 2-step NS iter & norm-match ($\text{rms}\rightarrow\text{rms}$)
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M^\dagger$,
    #                                                                                                                                                                                    r'where $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    # muon_2step_precond50_norm_jb_target_glbavgmomentum_dual=        (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_rawg_dual', backend_steps=2,
    #                                                                                    precondition_backend='newtonschulz5_sched10', precondition_backend_steps=10,
    #                                                                                    precondition_kind='min_dim', compute_precondition_freq=50, precondition_beta2=0.7),           [r'muon w/ 2-step preconditioned NS iter & norm-match ($\text{rms}\rightarrow\text{rms}$)',
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M^\dagger$,',
    #                                                                                                                                                                                    r'where $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    muon_pre_ns_norm_jb_target_glbavgmomentum_dual=                 (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_rawg_dual', momentum_kind='pre_ns'),          r'''muon w/ pre-ns ema & modula-like dual-norm-match ($\text{rms}\rightarrow\text{rms}$, avg over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                    '''),
    muon_post_norm_scale_nesterov_norm_jb_target_glbavggrad_dual=    (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_rawg_dual', momentum_kind='post_norm_scale_nesterov'),  r'''muon w/ dual-modula-norm-weighted nesterov update ($\text{rms}\rightarrow\text{rms}$, avg over $W_i$)
                                                                                                                                                                                          $\text{grad}^0_i := \text{NS}(\text{grad}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$
                                                                                                                                                                                          $\Delta W_i := \text{nesterov}(||\text{grad}||_M^\dagger \frac{\text{grad}^0_i}{||\text{grad}^0_i||_{\text{rms}\rightarrow\text{rms}}})$
                                                                                                                                                                                    '''),


    muon_norm_rms_target_glbmaxmomentum=                            (functools.partial(Muon, norm_kind='rms', target_norm='globalmax_rawg'),                                      r'''muon w/ modula-like norm-match (approx rms, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_\text{rms}$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),
    muon_norm_rms_exact_target_glbmaxmomentum=                      (functools.partial(Muon, norm_kind='rms_exact', target_norm='globalmax_rawg'),                                r'''muon w/ modula-like norm-match (rms, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_\text{rms}$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),
    muon_norm_fro_target_glbmaxmomentum=                            (functools.partial(Muon, norm_kind='fro', target_norm='globalmax_rawg'),                                      r'''muon w/ modula-like norm-match (frobenius, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_fro_exact_target_glbmaxmomentum=                      (functools.partial(Muon, norm_kind='fro_exact', target_norm='globalmax_rawg'),                                r'''muon w/ modula-like norm-match (frobenius, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_spec_target_glbmaxmomentum=                           (functools.partial(Muon, norm_kind='spectral', target_norm='globalmax_rawg'),                                 r'''muon w/ modula-like norm-match (spectral, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_*$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                    '''),
    muon_norm_jb_target_glbmaxmomentum=                             (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_rawg'),                                   r'''muon w/ modula-like norm-match ($\text{rms}\rightarrow\text{rms}$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}$
                                                                                                                                                                                          $\Delta W_i := ||\mathcal{W}||_M \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                    '''),

    muon_norm_rms_target_glbmaxmomentum_dual=                        (functools.partial(Muon, norm_kind='rms', target_norm='globalmax_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match (approx rms, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_\text{rms}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),
    muon_norm_rms_exact_target_glbmaxmomentum_dual=                  (functools.partial(Muon, norm_kind='rms_exact', target_norm='globalmax_rawg_dual'),                            r'''muon w/ modula-like dual-norm-match (rms, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_\text{rms}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_\text{rms}}$
                                                                                                                                                                                    '''),
    muon_norm_fro_target_glbmaxmomentum_dual=                        (functools.partial(Muon, norm_kind='fro', target_norm='globalmax_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match (frobenius, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_F^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_fro_exact_target_glbmaxmomentum_dual=                  (functools.partial(Muon, norm_kind='fro_exact', target_norm='globalmax_rawg_dual'),                            r'''muon w/ modula-like dual-norm-match (frobenius, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_F^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_F}$
                                                                                                                                                                                    '''),
    muon_norm_spec_target_glbmaxmomentum_dual=                       (functools.partial(Muon, norm_kind='spectral', target_norm='globalmax_rawg_dual'),                                 r'''muon w/ modula-like dual-norm-match (spectral, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_*^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_*}$
                                                                                                                                                                                    '''),
    muon_norm_jb_target_glbmaxmomentum_dual=                        (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_rawg_dual'),                              r'''muon w/ modula-like dual-norm-match ($\text{rms}\rightarrow\text{rms}$, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{nesterov}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                    '''),
    # muon_2step_norm_jb_target_glbmaxmomentum_dual=                   (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_rawg_dual', backend_steps=2),            [r'muon w/ 2-step NS iter & norm-match ($\text{rms}\rightarrow\text{rms}$)',
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
    #                                                                                                                                                                                    r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    # muon_2step_precond50_norm_jb_target_glbmaxmomentum_dual=        (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_rawg_dual', backend_steps=2,
    #                                                                                    precondition_backend='newtonschulz5_sched10', precondition_backend_steps=10,
    #                                                                                    precondition_kind='min_dim', compute_precondition_freq=50, precondition_beta2=0.7),           [r'muon w/ 2-step preconditioned NS iter & norm-match ($\text{rms}\rightarrow\text{rms}$)',
    #                                                                                                                                                                                    r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
    #                                                                                                                                                                                    r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    muon_pre_ns_norm_jb_target_glbmaxmomentum_dual=                 (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_rawg_dual', momentum_kind='pre_ns'),          r'''muon w/ pre-ns ema & modula-like dual-norm-match ($\text{rms}\rightarrow\text{rms}$, max over $W_i$)
                                                                                                                                                                                          $\text{momentum}_i := \text{ema}(\text{grad}_i)$
                                                                                                                                                                                          $\text{momentum}^0_i := \text{NS}(\text{momentum}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$
                                                                                                                                                                                          $\Delta W_i := ||\text{momentum}||_M^\dagger \frac{\text{momentum}^0_i}{||\text{momentum}^0_i||_{\text{rms}\rightarrow\text{rms}}}$
                                                                                                                                                                                    '''),
    muon_post_norm_scale_nesterov_norm_jb_target_glbmaxgrad_dual=    (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_rawg_dual', momentum_kind='post_norm_scale_nesterov'),  r'''muon w/ dual-modula-norm-weighted nesterov update ($\text{rms}\rightarrow\text{rms}$, max over $W_i$)
                                                                                                                                                                                          $\text{grad}^0_i := \text{NS}(\text{grad}_i)$
                                                                                                                                                                                          $||\mathcal{W}||_M^\dagger := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$
                                                                                                                                                                                          $\Delta W_i := \text{nesterov}(||\text{grad}||_M^\dagger \frac{\text{grad}^0_i}{||\text{grad}^0_i||_{\text{rms}\rightarrow\text{rms}}})$
                                                                                                                                                                                    '''),

    # muon_renorm_fro=                                          (functools.partial(Muon, renormalize='momentum', renorm_kind='fro'),                                                           [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_F$ match $||\text{momentum}||_F$)']),
    # muon_renorm_spec=                                         (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral'),                                                      [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_*$ match $||\text{momentum}||_*$)']),

    # muon_renorm_glbsfro=                                      (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='fro'),                                                 [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbmfro=                                      (functools.partial(Muon, renormalize='globalmax_rawg', renorm_kind='fro'),                                                 [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbsspec=                                     (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='spectral'),                                            [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_*$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbmspec=                                     (functools.partial(Muon, renormalize='globalmax_rawg', renorm_kind='spectral'),                                            [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_*$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_noscale=                                             (functools.partial(Muon, scale=None),                                                                                          [r'muon no scale']),
    # muon_jbscale=                                             (functools.partial(Muon, scale='jxbz'),                                                                                        [r'muon scale to unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$*$||\Delta W||_*$',
    #                                                                                                                                                                                           r'(default is unit $||\Delta W||_\text{RMS}$)']),

    # muon_renorm_fro_noscale=                                  (functools.partial(Muon, renormalize='momentum', renorm_kind='fro', scale=None),                                               [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_F$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ no scale']),

    # muon_renorm_spec_noscale=                                 (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral', scale=None),                                          [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_*$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ no scale']),

    # muon_renorm_fro_jbscale=                                  (functools.partial(Muon, renormalize='momentum', renorm_kind='fro', scale='jxbz'),                                             [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_F$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_spec_jbscale=                                 (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral', scale='jxbz'),                                        [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_*$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbsfro_jbscale=                              (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='fro', scale='jxbz'),                                   [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbmfro_jbscale=                              (functools.partial(Muon, renormalize='globalmax_rawg', renorm_kind='fro', scale='jxbz'),                                   [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbsspec_jbscale=                             (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='spectral', scale='jxbz'),                              [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_*$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbmspec_jbscale=                             (functools.partial(Muon, renormalize='globalmax_rawg', renorm_kind='spectral', scale='jxbz'),                              [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_*$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),
)

for k in list(OPTIM_MAP.keys()):
    if 'muon' in k:
        if 'precond' in k:
            del OPTIM_MAP[k]
        elif '_1step' in k:
            del OPTIM_MAP[k]
        elif '_2step' in k:
            del OPTIM_MAP[k]
        elif '_3step' in k:
            del OPTIM_MAP[k]

for k in list(OPTIM_MAP.keys()):
    optim_ctor, desc = OPTIM_MAP[k]
    desc = '\n'.join([l.strip() for l in desc.split('\n') if l.strip()])
    OPTIM_MAP[k] = (optim_ctor, desc)

EQUIV_MAPS: Mapping[str, str] = dict(
    muon_5step='muon',
    muon_pre_ns_nesterov='muon',
    muon_norm_rms_target_unit='muon',
    muon_momentum095='muon',
    adam_b09='adam',
)


def should_rerun(optim_kind):
    should_rerun = False
    test_opt = OPTIM_MAP[optim_kind][0]([torch.empty(1, 1, requires_grad=True)])
    if test_opt.__class__ in {optim.Adam, DistributedShampoo}:
        should_rerun = True
    else:
        assert test_opt.__class__ is Muon
        should_rerun = test_opt.defaults['norm_kind'] in {'rms', 'rms_exact', 'jbnorm', 'jbnorm_exact', 'jbinvnorm', 'jbinvnorm_exact'}
    return should_rerun


if __name__ == '__main__':
    import sys
    # argv: [optim, lr, seed]
    optim_kind, lr, seed = sys.argv[1:]
    lr = float(lr)
    seed = int(seed)
    print(optim_kind, lr, seed)

    file = f'241018_300steps_bzs2048/orth_{optim_kind}_lr{lr:g}_seed{seed}.pth'

    # if should_rerun(optim_kind):
    #     print(f'rerunning {file}')
    # else:
    force_rerun = os.environ.get('FORCE_RERUN', '0') == '1'
    if os.path.exists(file) and not force_rerun:
        print(f'skipping {file}')
        sys.exit()

    def run():
        if optim_kind in EQUIV_MAPS:
            actual_optim_kind = EQUIV_MAPS[optim_kind]
            actual_file = f'./orth_{actual_optim_kind}_lr{lr:g}_seed{seed}.pth'
            # check if a link already exists
            if os.path.islink(file):
                os.remove(file)
            os.symlink(actual_file, file)
            print(f'linked {actual_file} to {file}')
            return

        torch.manual_seed(seed + 21436)
        torch.cuda.manual_seed(seed + 21436)
        np.random.seed(seed + 21436)
        random.seed(seed + 21436)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f'training {file}')
        model = make_model()
        model.load_state_dict(torch.load('orth_init_weights.pth', weights_only=False)[seed])
        res = train_mnist(model, opt=OPTIM_MAP[optim_kind][0], w_save_key=None, lr=lr, nsteps=500, log_nsteps=5, batch_size=2048)
        torch.save(res._asdict(), file)
        print(f'saved {file}')

    with open(file + '.running', 'w') as f:
        run()
        f.write('0')
