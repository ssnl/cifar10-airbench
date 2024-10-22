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
import random


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
    rawg0: torch.Tensor                  # rawg^0
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

    def _compute_norm(self, tensor_kind: str, norm_kind: str, dual: bool = False):
        if dual:
            dual_norm_kind = dict(
                spectral='nuclear',
                nuclear='spectral',
                jbnorm='jbnuclear',
                jbnuclear='jbnorm',
                spectral_exact='nuclear_exact',
                nuclear_exact='spectral_exact',
                jbnorm_exact='jbnuclear_exact',
                jbnuclear_exact='jbnorm_exact',
                fro='fro',
                fro_exact='fro_exact',
                rms='rms',
                rms_exact='rms_exact',
            ).get(norm_kind, None)
            if dual_norm_kind is None:
                raise ValueError(f"dual norm kind {norm_kind} not supported")
            return self(tensor_kind, dual_norm_kind, dual=False)

        assert tensor_kind in ('rawg', 'g', 'grad')
        if norm_kind in {'rms_exact', 'rms'}:
            denom = self.rawg.numel() ** 0.5
            return self(tensor_kind, norm_kind.replace('rms', 'fro'), dual=dual) / denom
        elif norm_kind == 'fro_exact':
            if tensor_kind == 'rawg':
                return self.rawgnorm_fro
            elif tensor_kind == 'g':
                return self.g.norm()
            else:
                return self.grad.norm()
        elif norm_kind == 'fro':
            if tensor_kind == 'rawg':
                return self.rawgnorm_fro
            elif tensor_kind == 'g':
                assert self.zeropower_backend not in ('svd', 'sign'), "fro norm of g not supported for svd or sign"
                assert self.momentum_kind not in {'post_ns', 'post_ns_nesterov'}, "fro norm of g not supported for post-ns"
                # currently have ||DW||_fro     ~= sqrt(min(fan_in, fan_out))
                return self.min_fan**0.5
            else:
                return self.grad.norm()
        elif norm_kind in {'jbnorm', 'jbnorm_exact'}:
            # ||W|| := sqrt(fan_out / fan_in) * ||W||_2
            scale = (self.fan_out / self.fan_in)**0.5
            return self(tensor_kind, norm_kind.replace('jbnorm', 'spectral'), dual=dual) * scale
        elif norm_kind in ('spectral', 'spectral_exact'):
            # initialize power iterations on rawg (momentum/grad)
            # ref: https://github.com/jxbz/modula/blob/e274a352551ec4c6055b7fc0086db7a516863578/modula/atom.py#L32
            #      https://github.com/pytorch/pytorch/blob/d7e0e1dbc453bac099f747dfb65ad75767c3e1d7/torch/nn/utils/spectral_norm.py#L96
            if tensor_kind == 'g':
                assert self.zeropower_backend not in ('svd', 'sign'), "spectral norm of g not supported for svd or sign"
                assert self.momentum_kind not in {'post_ns', 'post_ns_nesterov'}, "spectral norm of g not supported for post-ns"
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
        elif norm_kind in {'jbnuclear', 'jbnuclear_exact'}:
            scale = (self.fan_out / self.fan_in)**0.5
            return self(tensor_kind, norm_kind.replace('jbnuclear', 'nuclear'), dual=dual) * scale
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
                assert self.momentum_kind not in {'post_ns', 'post_ns_nesterov'}, "nuclear (est) norm of g not supported for post-ns"
                return self(tensor_kind, norm_kind.replace('nuclear', 'fro'), dual=dual)
            else:
                assert False, f"{norm_kind} not implemented for tensor_kind {tensor_kind}"
        else:
            assert False, f"unknown norm kind {norm_kind}"


@torch.compile
def _right_preconditioner_from_zerothpower(g, g0, sqrt_dim: float, dtype=torch.float32, eps=1e-7):
    # return V S-1 V.T
    vsv = (g0.T @ g).to(dtype)
    vsv = vsv / vsv.norm() * sqrt_dim
    vsv.diagonal(dim1=-2, dim2=-1).add_(eps)
    L, info = torch.linalg.cholesky_ex(vsv)
    if info.item() != 0:
        raise RuntimeError(f"cholesky_ex failed with info {info}")
    inv = torch.cholesky_inverse(L).to(g.dtype)
    return inv / inv.norm() * sqrt_dim

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
    usu = usu / usu.norm() * sqrt_dim
    usu.diagonal(dim1=-2, dim2=-1).add_(eps)
    L, info = torch.linalg.cholesky_ex(usu)
    if info.item() != 0:
        raise RuntimeError(f"cholesky_ex failed with info {info}")
    inv = torch.cholesky_inverse(L).to(g.dtype)
    return inv / inv.norm() * sqrt_dim

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
                 backend_steps=5, norm_kind='rms', target_norm='unit', eps=1e-7, compute_precondition_freq=20, precondition_kind=None):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, momentum_kind=momentum_kind, backend=backend, backend_steps=backend_steps,
                        norm_kind=norm_kind, target_norm=target_norm, eps=eps, compute_precondition_freq=compute_precondition_freq, precondition_kind=precondition_kind)
        assert momentum_kind in {'pre_ns', 'pre_ns_nesterov', 'post_ns', 'post_ns_nesterov', None}
        assert precondition_kind in {'left', 'left_lstsq' 'right', 'min_dim', None}
        super().__init__(params, defaults)

    def _apply_momentum(self, state, x: torch.Tensor, momentum, *, is_nesterov):
        # m = beta1 * m + g
        # g = g + m * beta1
        # ----
        # equiv:
        # g = g + beta1**2 * m + beta1 * g
        #   = (1+beta1) g + beta1**2 m
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(x)
        buf = state['momentum_buffer']
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
            zeropower_backend = zeropower_backends[group['backend']]

            for p in group['params']:
                # rawg: grad or momentum grad
                # g: after zeropower

                rawg = p.grad
                if rawg is None:
                    continue

                state = self.state[p]
                if 'stept' not in state:
                    state['stept'] = 0
                state['stept'] += 1
                stept = state['stept']

                if group['momentum_kind'] in {'pre_ns', 'pre_ns_nesterov'}:
                    rawg = self._apply_momentum(state, rawg, momentum, is_nesterov=group['momentum_kind'] == 'pre_ns_nesterov')

                # apply preconditioner
                arg_precondition_kind = group['precondition_kind']
                if arg_precondition_kind == 'min_dim':
                    precondition_kind = 'left' if p.shape[0] < p.shape[1] else 'right'
                else:
                    precondition_kind = arg_precondition_kind
                if precondition_kind is not None and (preconditioner:= state.get('preconditioner', None)) is not None:
                    if precondition_kind == 'left':
                        rawg = preconditioner @ rawg
                    elif precondition_kind == 'right':
                        rawg = rawg @ preconditioner
                    else:
                        assert False, f"unknown precondition_kind {group['precondition_kind']}"

                rawgnorm_fro = rawg.norm()
                rawg0 = zeropower_backend(rawg, steps=group['backend_steps'], dtype=rawg.dtype, G_fro=rawgnorm_fro)

                # update preconditioner
                if precondition_kind is not None and stept > 0 and stept % group['compute_precondition_freq'] == 0:
                    assert group['backend'] not in {'sgd', 'sign'}, "preconditioner not supported for sgd or sign"
                    # here we don something "hacky" to pick eps
                    # let's assume that rawg0 has singular values in [0.95, 1.05]
                    # rawg/rawgnorm_fro has singular values in [0, 1]
                    # now the preconditioner is generally (rawg0.T @ rawg + eps I)^{-1}
                    # note that scaling preconditioner doesn't matter for the 0th power
                    # so we can also do (rawg0.T @ rawg / C1 + eps I)^{-1} / C2
                    if precondition_kind == 'left':
                        new_preconditioner = left_preconditioner_from_zerothpower_with_retry(rawg, rawg0, eps=1e-3)
                    elif precondition_kind == 'right':
                        new_preconditioner = right_preconditioner_from_zerothpower_with_retry(rawg, rawg0, eps=1e-3)
                    if new_preconditioner is None and (preconditioner:= state.get('preconditioner', None)) is not None:
                        # regress to idt
                        preconditioner.lerp_(torch.eye(preconditioner.shape[0], device=preconditioner.device, dtype=preconditioner.dtype), 1 - group['beta2'] ** group['compute_precondition_freq'])
                    else:
                        state['preconditioner'] = new_preconditioner

                if group['momentum_kind'] in {'post_ns', 'post_ns_nesterov'}:
                    g = self._apply_momentum(state, rawg0, momentum, is_nesterov=group['momentum_kind'] == 'post_ns_nesterov')
                else:
                    g = rawg0

                state['last_update'] = dict(grad=p.grad, rawg=rawg, rawg0=rawg0, g=g, rawgnorm_fro=rawgnorm_fro)

            # renormalize update
            norms = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                norms[p] = NormInterface(
                    state,
                    zeropower_backend=group['backend'],
                    momentum_kind=group['momentum_kind'],
                    eps=eps,
                    **state['last_update'],
                )

            norm_kind = group['norm_kind']
            target_norm = group['target_norm']

            if target_norm == 'unit':
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

                elif target_norm == 'ema_momentum_norm':
                    # target_norm = ema(||momentum||)
                    for p, norm_interface in norms.items():
                        state = self.state[p]
                        if 'ema_momentum_norm' not in state:
                            state['ema_momentum_norm'] = p.grad.new_zeros(())
                        torch.lerp(state['ema_momentum_norm'], norm_interface('rawg', norm_kind, dual=use_dual), 1 - group['beta2'], out=state['ema_momentum_norm'])
                        self.state[p]['last_update']['target_norm'] = state['ema_momentum_norm'] / (1 - group['beta2']**stept)

                elif target_norm == 'ema_grad_norm2_sqrt':
                    # target_norm = ema(||grad||^2)^0.5
                    for p, norm_interface in norms.items():
                        state = self.state[p]
                        if 'ema_grad_norm2' not in state:
                            state['ema_grad_norm2'] = p.grad.new_zeros(())
                        torch.lerp(state['ema_grad_norm2'], norm_interface('grad', norm_kind, dual=use_dual)**2, 1 - group['beta2'], out=state['ema_grad_norm2'])
                        self.state[p]['last_update']['target_norm'] = (state['ema_grad_norm2'] / (1 - group['beta2']**stept))**0.5

                elif target_norm == 'ema_momentum_norm2_sqrt':
                    # target_norm = ema(||momentum||^2)^0.5
                    for p, norm_interface in norms.items():
                        state = self.state[p]
                        if 'ema_momentum_norm' not in state:
                            state['ema_momentum_norm2'] = p.grad.new_zeros(())
                        torch.lerp(state['ema_momentum_norm2'], norm_interface('rawg', norm_kind, dual=use_dual)**2, 1 - group['beta2'], out=state['ema_momentum_norm2'])
                        self.state[p]['last_update']['target_norm'] = (state['ema_momentum_norm2'] / (1 - group['beta2']**stept))**0.5

                elif target_norm == 'momentum':
                    # target_norm = ||momentum||
                    for p, norm_interface in norms.items():
                        self.state[p]['last_update']['target_norm'] = norm_interface('rawg', norm_kind, dual=use_dual)

                else:
                    if target_norm == 'globalavg_momentum':
                        target_norm = (
                            sum(norm_interface('rawg', norm_kind, dual=use_dual) for _, norm_interface in norms.items())
                            /
                            len(norms)
                        )
                    elif target_norm == 'globalmax_momentum':
                        target_norm = max(norm_interface('rawg', norm_kind, dual=use_dual) for _, norm_interface in norms.items())
                    else:
                        assert False, f"unknown target_norm {group['target_norm']}"

                    for p, norm_interface in norms.items():
                        self.state[p]['last_update']['target_norm'] = target_norm

            for p, norm_interface in norms.items():
                state = self.state[p]
                last_update = state['last_update']
                scale = last_update['target_norm'] / norm_interface('g', norm_kind, dual=False)  # see anthology proposition 1. unit norm, scale to ||g||^dagger
                p.data.add_(norm_interface.g, alpha=-lr * scale)

Result = namedtuple('Result', ['steps', 'train_accs', 'eval_accs', 'model_ws', 'state_dict'])

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
        optimizers.extend([
            opt([p for p in model.parameters() if len(p.data.shape) == 2], lr=lr),
            optim.Adam([p for p in model.parameters() if len(p.data.shape) != 2], lr=lr),
        ])
    elif opt == 'adam':
        optimizers.append(
            optim.Adam(model.parameters(), lr=lr)  # Using Adam optimizer
        )
    elif opt == 'adam_b095':
        optimizers.append(
            optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.999))
        )
    elif opt == 'adam_b0995':
        optimizers.append(
            optim.Adam(model.parameters(), lr=lr, betas=(0.995, 0.999))
        )
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
        data, target = data.to(device, dtype), target.to(device)

        model.zero_grad()
        output = model(data.flatten(1))
        loss = F.cross_entropy(output, target)
        (loss * loss_scale).backward()

        for optimizer in optimizers:
            optimizer.step()

        if post_step_callback is not None:
            post_step_callback(step, model, optimizers, data, target, loss)

        if step % log_nsteps == 0:
            print(f"Step {step}/{nsteps}, Loss: {loss.item():.4f}")
            record(step)

        if step == nsteps:
            break

    print("Training completed!")

    return Result(steps, train_accs, eval_accs, model_ws, model.state_dict())


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
    adam=                                                           ('adam',                                                                                                          [r'adam',
                                                                                                                                                                                       r'(default $\beta_1$=0.9)']),
    adam_b095=                                                      ('adam_b095',                                                                                                     [r'adam',
                                                                                                                                                                                       r'($\beta_1$=0.9$\rightarrow$0.95)']),
    adam_b0995=                                                     ('adam_b0995',                                                                                                    [r'adam',
                                                                                                                                                                                       r'($\beta_1$=0.9$\rightarrow$0.995)']),
    muon=                                                           (functools.partial(Muon, backend='newtonschulz5'),                                                                [r'muon']),

    muon_pre_ns=                                                    (functools.partial(Muon, momentum_kind='pre_ns'),                                                                 [r'muon w pre-ns ema']),

    muon_post_ns=                                                   (functools.partial(Muon, momentum_kind='post_ns'),                                                                [r'muon w post-ns ema']),

    muon_pre_ns_nesterov=                                           (functools.partial(Muon, momentum_kind='pre_ns_nesterov'),                                                        [r'muon w pre-ns nesterov-type update']),

    muon_post_ns_nesterov=                                          (functools.partial(Muon, momentum_kind='post_ns_nesterov'),                                                       [r'muon w post-ns nesterov-type update']),


    muon_sgd=                                                       (functools.partial(Muon, backend='sgd'),                                                                          [r'SGD',
                                                                                                                                                                                       r'(i.e., muon w/o orthogonalization)']),

    muon_sign=                                                      (functools.partial(Muon, backend='sign'),                                                                         [r'sign-SGD',
                                                                                                                                                                                       r'(i.e., muon w/ sign instead of orthogonalization)']),

    muon_proper=                                                    (functools.partial(Muon, backend='newtonschulz5_proper'),                                                         [r'muon w naive simple cubic NS iter',
                                                                                                                                                                                       r'(still 5 steps)']),
    muon_sched5=                                                    (functools.partial(Muon, backend='newtonschulz5_sched5', backend_steps=5),                                        [r'muon w scheduled 5-step NS iter',
                                                                                                                                                                                       r'(default$\rightarrow$naive)']),
    muon_sched8=                                                    (functools.partial(Muon, backend='newtonschulz5_sched8', backend_steps=8),                                        [r'muon w scheduled 8-step NS iter',
                                                                                                                                                                                       r'(default$\rightarrow$naive)']),
    muon_sched10=                                                   (functools.partial(Muon, backend='newtonschulz5_sched10', backend_steps=10),                                      [r'muon w scheduled 10-step NS iter',
                                                                                                                                                                                       r'(default$\rightarrow$naive)']),
    muon_sched14=                                                   (functools.partial(Muon, backend='newtonschulz5_sched14', backend_steps=14),                                      [r'muon w scheduled 14-step NS iter',
                                                                                                                                                                                       r'(default$\rightarrow$naive)']),

    muon_momentum099=                                               (functools.partial(Muon, backend='newtonschulz5', momentum=0.99),                                                 [r'muon momentum 0.95$\rightarrow$0.99']),
    muon_momentum095=                                               (functools.partial(Muon, backend='newtonschulz5', momentum=0.95),                                                 [r'muon momentum 0.95 (default)']),
    muon_momentum09=                                                (functools.partial(Muon, backend='newtonschulz5', momentum=0.9),                                                  [r'muon momentum 0.95$\rightarrow$0.9']),
    muon_momentum085=                                               (functools.partial(Muon, backend='newtonschulz5', momentum=0.85),                                                 [r'muon momentum 0.95$\rightarrow$0.85']),
    muon_momentum08=                                                (functools.partial(Muon, backend='newtonschulz5', momentum=0.8),                                                  [r'muon momentum 0.95$\rightarrow$0.8']),
    muon_no_momentum=                                               (functools.partial(Muon, backend='newtonschulz5', momentum_kind=None),                                            [r'muon no momentum']),

    muon_norm_rms_target_unit=                                      (functools.partial(Muon, norm_kind='rms', target_norm='unit'),                                                    [r'muon (default, rms)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_\text{rms}$ to be 1)']),
    muon_norm_fro_target_unit=                                      (functools.partial(Muon, norm_kind='fro', target_norm='unit'),                                                    [r'muon (frobenius)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to be 1)']),
    muon_norm_fro_exact_target_unit=                                (functools.partial(Muon, norm_kind='fro_exact', target_norm='unit'),                                              [r'muon (frobenius, exactly computed on NS outputs)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to be 1)']),
    muon_norm_spec_target_unit=                                     (functools.partial(Muon, norm_kind='spectral', target_norm='unit'),                                               [r'muon (spectral)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_2$ to be 1)']),
    muon_norm_jb_target_unit=                                       (functools.partial(Muon, norm_kind='jbnorm', target_norm='unit'),                                                 [r'muon ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to be 1)']),
    muon_pre_ns_norm_jb_target_unit=                                (functools.partial(Muon, norm_kind='jbnorm', target_norm='unit', momentum_kind='pre_ns'),                         [r'muon w pre-ns ema & ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to be 1)']),


    muon_norm_rms_target_momentum=                                  (functools.partial(Muon, norm_kind='rms', target_norm='momentum'),                                                [r'muon norm-match (rms)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_\text{rms}$ to match $||\text{momentum}_i||_\text{rms}$)']),
    muon_norm_fro_target_momentum=                                  (functools.partial(Muon, norm_kind='fro', target_norm='momentum'),                                                [r'muon norm-match (frobenius)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match $||\text{momentum}_i||_F$)']),
    muon_norm_fro_exact_target_momentum=                            (functools.partial(Muon, norm_kind='fro_exact', target_norm='momentum'),                                          [r'muon norm-match (frobenius, exactly computed on NS outputs)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match $||\text{momentum}_i||_F$)']),
    muon_norm_spec_target_momentum=                                 (functools.partial(Muon, norm_kind='spectral', target_norm='momentum'),                                           [r'muon norm-match (spectral)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_2$ to match $||\text{momentum}_i||_2$)']),
    muon_norm_jb_target_momentum=                                   (functools.partial(Muon, norm_kind='jbnorm', target_norm='momentum'),                                             [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}$']),
    muon_norm_jb_target_momentum_dual=                              (functools.partial(Muon, norm_kind='jbnorm', target_norm='momentum_dual'),                                         [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    muon_pre_ns_norm_jb_target_momentum_dual=                       (functools.partial(Muon, norm_kind='jbnorm', target_norm='momentum_dual', momentum_kind='pre_ns'),                [r'muon w pre-ns ema & norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $||\text{momentum}_i^\text{ema}||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),


    muon_norm_rms_target_ema_grad_norm2_sqrt=                       (functools.partial(Muon, norm_kind='rms', target_norm='ema_grad_norm2_sqrt'),                                     [r'muon norm-match (rms)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_\text{rms}$ to match $\text{ema}(||\text{momentum}_i||_{\text{rms}}^2)^{1/2}$)']),
    muon_norm_fro_target_ema_grad_norm2_sqrt=                       (functools.partial(Muon, norm_kind='fro', target_norm='ema_grad_norm2_sqrt'),                                     [r'muon norm-match (frobenius)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match $\text{ema}(||\text{momentum}_i||_F^2)^{1/2}$)']),
    muon_norm_fro_exact_target_ema_grad_norm2_sqrt=                 (functools.partial(Muon, norm_kind='fro_exact', target_norm='ema_grad_norm2_sqrt'),                                [r'muon norm-match (frobenius, exactly computed on NS outputs)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match $\text{ema}(||\text{momentum}_i||_F^2)^{1/2}$)']),
    muon_norm_spec_target_ema_grad_norm2_sqrt=                      (functools.partial(Muon, norm_kind='spectral', target_norm='ema_grad_norm2_sqrt'),                                [r'muon norm-match (spectral)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_2$ to match $\text{ema}(||\text{momentum}_i||_2^2)^{1/2}$)']),
    muon_norm_jb_target_ema_grad_norm2_sqrt=                        (functools.partial(Muon, norm_kind='jbnorm', target_norm='ema_grad_norm2_sqrt'),                                  [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $\text{ema}(||\text{momentum}_i||_{\text{rms}}^2)^{1/2}$)']),
    muon_norm_jb_target_ema_grad_norm2_sqrt_dual=                   (functools.partial(Muon, norm_kind='jbnorm', target_norm='ema_grad_norm2_sqrt_dual'),                             [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $\text{ema}((||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger)^2)^{1/2}$)']),
    muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual=            (functools.partial(Muon, norm_kind='jbnorm', target_norm='ema_grad_norm2_sqrt_dual', momentum_kind='pre_ns'),     [r'muon w pre-ns ema & norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $\text{ema}((||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}^\dagger)^2)^{1/2}$)']),


    muon_norm_rms_target_glbavgmomentum=                            (functools.partial(Muon, norm_kind='rms', target_norm='globalavg_momentum'),                                      [r'muon norm-match (rms)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_\text{rms}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_\text{rms}$)']),
    muon_norm_fro_target_glbavgmomentum=                            (functools.partial(Muon, norm_kind='fro', target_norm='globalavg_momentum'),                                      [r'muon norm-match (frobenius)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_F$)']),
    muon_norm_fro_exact_target_glbavgmomentum=                      (functools.partial(Muon, norm_kind='fro_exact', target_norm='globalavg_momentum'),                                [r'muon norm-match (frobenius, exactly computed on NS outputs)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_F$)']),
    muon_norm_spec_target_glbavgmomentum=                           (functools.partial(Muon, norm_kind='spectral', target_norm='globalavg_momentum'),                                 [r'muon norm-match (spectral)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_2$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_2$)']),
    muon_norm_jb_target_glbavgmomentum=                             (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_momentum'),                                   [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}$)']),
    muon_norm_jb_target_glbavgmomentum_dual=                        (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_momentum_dual'),                              [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    muon_pre_ns_norm_jb_target_glbavgmomentum_dual=                 (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_momentum_dual', momentum_kind='pre_ns'),      [r'muon w pre-ns ema & norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),

    muon_norm_rms_target_glbmaxmomentum=                            (functools.partial(Muon, norm_kind='rms', target_norm='globalmax_momentum'),                                      [r'muon norm-match (rms)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_\text{rms}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_\text{rms}$)']),
    muon_norm_fro_target_glbmaxmomentum=                            (functools.partial(Muon, norm_kind='fro', target_norm='globalmax_momentum'),                                      [r'muon norm-match (frobenius)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$)']),
    muon_norm_fro_exact_target_glbmaxmomentum=                      (functools.partial(Muon, norm_kind='fro_exact', target_norm='globalmax_momentum'),                                [r'muon norm-match (frobenius, exactly computed on NS outputs)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_F$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$)']),
    muon_norm_spec_target_glbmaxmomentum=                           (functools.partial(Muon, norm_kind='spectral', target_norm='globalmax_momentum'),                                 [r'muon norm-match (spectral)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_2$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_2$)']),
    muon_norm_jb_target_glbmaxmomentum=                             (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_momentum'),                                   [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}$)']),
    muon_norm_jb_target_glbmaxmomentum_dual=                        (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_momentum_dual'),                              [r'muon norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),
    muon_pre_ns_norm_jb_target_glbmaxmomentum_dual=                 (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_momentum_dual', momentum_kind='pre_ns'),      [r'muon w pre-ns ema & norm-match ($\text{rms}\rightarrow\text{rms}$)',
                                                                                                                                                                                       r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                                                                       r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}^\dagger$)']),

    # muon_renorm_fro=                                          (functools.partial(Muon, renormalize='momentum', renorm_kind='fro'),                                                           [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_F$ match $||\text{momentum}||_F$)']),
    # muon_renorm_spec=                                         (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral'),                                                      [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_2$ match $||\text{momentum}||_2$)']),

    # muon_renorm_glbsfro=                                      (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='fro'),                                                 [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbmfro=                                      (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='fro'),                                                 [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbsspec=                                     (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='spectral'),                                            [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbmspec=                                     (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='spectral'),                                            [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_noscale=                                             (functools.partial(Muon, scale=None),                                                                                          [r'muon no scale']),
    # muon_jbscale=                                             (functools.partial(Muon, scale='jxbz'),                                                                                        [r'muon scale to unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$*$||\Delta W||_2$',
    #                                                                                                                                                                                           r'(default is unit $||\Delta W||_\text{RMS}$)']),

    # muon_renorm_fro_noscale=                                  (functools.partial(Muon, renormalize='momentum', renorm_kind='fro', scale=None),                                               [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_F$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ no scale']),

    # muon_renorm_spec_noscale=                                 (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral', scale=None),                                          [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_2$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ no scale']),

    # muon_renorm_fro_jbscale=                                  (functools.partial(Muon, renormalize='momentum', renorm_kind='fro', scale='jxbz'),                                             [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_F$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_spec_jbscale=                                 (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral', scale='jxbz'),                                        [r'muon renorm',
    #                                                                                                                                                                                           r'($||\Delta W||_2$ match $||\text{momentum}||$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbsfro_jbscale=                              (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='fro', scale='jxbz'),                                   [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbmfro_jbscale=                              (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='fro', scale='jxbz'),                                   [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbsspec_jbscale=                             (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='spectral', scale='jxbz'),                              [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbmspec_jbscale=                             (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='spectral', scale='jxbz'),                              [r'muon renorm',
    #                                                                                                                                                                                           r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                                                                           r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                                                                           r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),
)

EQUIV_MAPS: Mapping[str, str] = dict(
    muon_pre_ns_nesterov='muon',
    muon_norm_rms_target_unit='muon',
    muon_momentum095='muon',
)


if __name__ == '__main__':
    import sys
    # argv: [optim, lr, seed]
    optim_kind, lr, seed = sys.argv[1:]
    lr = float(lr)
    seed = int(seed)
    print(optim_kind, lr, seed)

    file = f'241018_300steps_bzs2048/orth_{optim_kind}_lr{lr:g}_seed{seed}.pth'

    if os.path.exists(file):
        print(f'skipping {file}')
        sys.exit()

    def run():
        if optim_kind in EQUIV_MAPS:
            actual_optim_kind = EQUIV_MAPS[optim_kind]
            actual_file = f'241018_300steps_bzs2048/orth_{actual_optim_kind}_lr{lr:g}_seed{seed}.pth'
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
