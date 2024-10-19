from typing import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import functools
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

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7, dtype=torch.bfloat16,
                                 abc: torch.Tensor = torch.tensor((3.4445, -4.7750,  2.0315)), normalize=True):
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
    if normalize:
        denom = G.norm() + eps
        X = G.to(dtype) / denom # ensure top singular value <= 1
    else:
        X = G.to(dtype)
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

def make_cached_func(fn: Callable[[], Any]) -> Callable[[], Any]:
    called = False
    res = None
    @functools.wraps(fn)
    def cached_fn():
        nonlocal called, res
        if not called:
            res = fn()
            called = True
        return res
    return cached_fn

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
    def __init__(self, params, lr=3e-4, momentum=0.95, beta2=0.999, nesterov=True, backend='newtonschulz5', backend_steps=5,
                 norm_kind='rms', target_norm='unit'):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, nesterov=nesterov, backend=backend, backend_steps=backend_steps,
                        norm_kind=norm_kind, target_norm=target_norm)
        super().__init__(params, defaults)

    def step(self):
        eps = 1e-7
        for group in self.param_groups:
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
                if group['nesterov']:
                    # m = beta1 * m + g
                    # g = g + m * beta1
                    # ----
                    # equiv:
                    # g = g + beta1**2 * m + beta1 * g
                    #   = (1+beta1) g + beta1**2 m
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(rawg)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(rawg)
                    rawg = rawg.add(buf, alpha=momentum)
                else:
                    rawg = rawg.clone()

                rawgnorm_fro = rawg.norm()
                g = rawg / (rawgnorm_fro + eps) # ensure top singular value <= 1
                g = zeropower_backend(g, steps=group['backend_steps'], dtype=g.dtype, normalize=False)

                # compute ||rawg|| and ||g|| under norm_kind
                if group['norm_kind'] == 'rms_exact':
                    # fro to rms
                    rawgnorm = rawgnorm_fro / (g.numel()**0.5)
                    gnorm = g.norm() / (g.numel()**0.5)
                elif group['norm_kind'] == 'rms':
                    # fro to rms
                    rawgnorm = rawgnorm_fro / (g.numel()**0.5)
                    # currently have ||DW||_fro     ~= sqrt(min(fan_in, fan_out))
                    #                ||DW||_rms_fro ~= 1/sqrt(max(fan_in, fan_out))
                    gnorm = 1 / max(g.size(0), g.size(1))**0.5
                elif group['norm_kind'] == 'fro':
                    rawgnorm = rawgnorm_fro
                    gnorm = g.norm()
                elif group['norm_kind'] in ('spectral', 'spectral_exact', 'jbnorm', 'jbnorm_exact'):
                    # initialize power iterations on rawg (momentum/grad)
                    # ref: https://github.com/jxbz/modula/blob/e274a352551ec4c6055b7fc0086db7a516863578/modula/atom.py#L32
                    if 'rawg_u' not in state:
                        state['rawg_u'] = F.normalize(torch.randn_like(rawg[0]), dim=0, eps=eps)
                        state['rawg_v'] = torch.empty_like(rawg[:, 0])
                        niter = 5
                    else:
                        niter = 1
                    u = state['rawg_u']
                    v = state['rawg_v']
                    for _ in range(niter):
                        torch.mv(rawg, u, out=v)
                        F.normalize(v, dim=0, eps=eps, out=v)
                        torch.mv(rawg.T, v, out=u)
                        F.normalize(u, dim=0, eps=eps, out=u)
                    rawgnorm = torch.dot(v, torch.mv(rawg, u))
                    if group['norm_kind'] == 'spectral':
                        # g should only have binary singular values, just assume 1! if it is 0, then scaling it with 1 is still 0
                        gnorm = 1
                    elif group['norm_kind'] == 'spectral_exact':
                        # g should only have binary singular values... having at least 1 non-zero means that frobenius norm is at least 1 = \sqrt{ \sum_i s_i^2 }
                        gnorm = torch.ge(g.norm(), 0.9, out=g.new_empty(()))
                    elif group['norm_kind'] == 'jbnorm':
                        # ||W|| := sqrt(fan_out / fan_in) * ||W||_2
                        gnorm = (g.size(0) / g.size(1))**0.5
                    elif group['norm_kind'] == 'jbnorm_exact':
                        gnorm = (g.size(0) / g.size(1))**0.5 * torch.ge(g.norm(), 0.9, out=g.new_empty(()))
                else:
                    assert False, f'unknown norm kind {group['renorm_kind']}'

                state['last_update'] = (g, rawgnorm, gnorm)

            # compute rawgnorm and gnorm
            rawgnorm_fn = lambda g, rawgnorm, gnorm: rawgnorm
            gnorm_fn = lambda g, rawgnorm, gnorm: gnorm

            if group['target_norm'] == 'unit':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.state[p]['last_update'] += (1, )
            elif group['target_norm'] == 'momentum':
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g, rawgnorm, gnorm = self.state[p]['last_update']
                    target_norm = rawgnorm_fn(g, rawgnorm, gnorm)
                    self.state[p]['last_update'] += (target_norm, )
            else:
                if group['target_norm'] == 'globalavg_momentum':
                    target_norm = (
                        sum(rawgnorm_fn(*self.state[p]['last_update']) for p in group['params'] if p.grad is not None)
                        /
                        sum(1 for p in group['params'] if p.grad is not None)
                    )
                elif group['target_norm'] == 'globalmax_momentum':
                    target_norm = max(rawgnorm_fn(*self.state[p]['last_update']) for p in group['params'] if p.grad is not None)
                elif group['target_norm'] is None:
                    target_norm = 1
                else:
                    assert False, f"unknown target_norm {group['target_norm']}"
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.state[p]['last_update'] += (target_norm, )

            for p in group['params']:
                if p.grad is None:
                    continue
                g, rawgnorm, gnorm, target_norm = self.state[p]['last_update']
                p.data.add_(g, alpha=-lr * (target_norm / gnorm))

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


# name -> (optim_cls, desc)
OPTIM_MAP: Mapping[str, Tuple[Union[str, Callable], List[str]]] = dict(
    adam=                                 ('adam',                                                                              [r'adam',
                                                                                                                                 r'(default $\beta_1$=0.9)']),
    adam_b095=                            ('adam_b095',                                                                         [r'adam_b095',
                                                                                                                                 r'($\beta_1$=0.9$\rightarrow$0.95)']),
    adam_b0995=                           ('adam_b0995',                                                                        [r'adam_b0995',
                                                                                                                                 r'($\beta_1$=0.9$\rightarrow$0.995)']),
    # muon=                                 (functools.partial(Muon, backend='newtonschulz5'),                                    [r'muon']),
    muon_sgd=                             (functools.partial(Muon, backend='sgd'),                                              [r'SGD',
                                                                                                                                 r'(i.e., muon w/o orthogonalization)']),

    muon_momentum08=                      (functools.partial(Muon, backend='newtonschulz5', momentum=0.8),                      [r'muon momentum 0.95$\rightarrow$0.8']),
    muon_momentum085=                     (functools.partial(Muon, backend='newtonschulz5', momentum=0.85),                     [r'muon momentum 0.95$\rightarrow$0.85']),
    muon_momentum09=                      (functools.partial(Muon, backend='newtonschulz5', momentum=0.9),                      [r'muon momentum 0.95$\rightarrow$0.9']),
    muon_momentum095=                     (functools.partial(Muon, backend='newtonschulz5', momentum=0.95),                     [r'muon momentum 0.95$\rightarrow$0.95']),
    muon_momentum099=                     (functools.partial(Muon, backend='newtonschulz5', momentum=0.99),                     [r'muon momentum 0.95$\rightarrow$0.99']),

    muon_no_momentum=                     (functools.partial(Muon, backend='newtonschulz5', nesterov=False),                    [r'muon no momentum']),
    muon_proper=                          (functools.partial(Muon, backend='newtonschulz5_proper'),                             [r'muon w naive simple cubic NS iter',
                                                                                                                                 r'(still 5 steps)']),
    muon_sched5=                          (functools.partial(Muon, backend='newtonschulz5_sched5', backend_steps=5),            [r'muon w scheduled 5-step NS iter',
                                                                                                                                 r'(default$\rightarrow$naive)']),
    muon_sched8=                          (functools.partial(Muon, backend='newtonschulz5_sched8', backend_steps=8),            [r'muon w scheduled 8-step NS iter',
                                                                                                                                 r'(default$\rightarrow$naive)']),
    muon_sched10=                         (functools.partial(Muon, backend='newtonschulz5_sched10', backend_steps=10),          [r'muon w scheduled 10-step NS iter',
                                                                                                                                 r'(default$\rightarrow$naive)']),
    muon_sched14=                         (functools.partial(Muon, backend='newtonschulz5_sched14', backend_steps=14),          [r'muon w scheduled 14-step NS iter',
                                                                                                                                 r'(default$\rightarrow$naive)']),

    muon_norm_rms_target_unit=            (functools.partial(Muon, norm_kind='rms', target_norm='unit'),                        [r'muon (default, rms, as if flattened as a vector)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_\text{rms}$ to be 1)']),
    muon_norm_fro_target_unit=            (functools.partial(Muon, norm_kind='fro', target_norm='unit'),                        [r'muon (frobenius)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_F$ to be 1)']),
    muon_norm_spec_target_unit=           (functools.partial(Muon, norm_kind='spectral', target_norm='unit'),                   [r'muon (spectral)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_2$ to be 1)']),
    muon_norm_jb_target_unit=             (functools.partial(Muon, norm_kind='jbnorm', target_norm='unit'),                     [r'muon ($\text{rms}\rightarrow	ext{rms}$, good norm)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to be 1)']),


    muon_norm_rms_target_momentum=        (functools.partial(Muon, norm_kind='rms', target_norm='momentum'),                    [r'muon norm-match (rms, as if flattened as a vector)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_\text{rms}$ to match $||\text{momentum}_i||_\text{rms}$)']),
    muon_norm_fro_target_momentum=        (functools.partial(Muon, norm_kind='fro', target_norm='momentum'),                    [r'muon norm-match (frobenius)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_F$ to match $||\text{momentum}_i||_F$)']),
    muon_norm_spec_target_momentum=       (functools.partial(Muon, norm_kind='spectral', target_norm='momentum'),               [r'muon norm-match (spectral)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_2$ to match $||\text{momentum}_i||_2$)']),
    muon_norm_jb_target_momentum=         (functools.partial(Muon, norm_kind='jbnorm', target_norm='momentum'),                 [r'muon norm-match ($\text{rms}\rightarrow	ext{rms}$, good norm)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match $||\text{momentum}_i||_{\text{rms}\rightarrow\text{rms}}$']),


    muon_norm_rms_target_glbavgmomentum=  (functools.partial(Muon, norm_kind='rms', target_norm='globalavg_momentum'),          [r'muon norm-match (rms, as if flattened as a vector)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_\text{rms}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_\text{rms}$)']),
    muon_norm_fro_target_glbavgmomentum=  (functools.partial(Muon, norm_kind='fro', target_norm='globalavg_momentum'),          [r'muon norm-match (frobenius)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_F$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_F$)']),
    muon_norm_spec_target_glbavgmomentum= (functools.partial(Muon, norm_kind='spectral', target_norm='globalavg_momentum'),     [r'muon norm-match (spectral)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_2$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_2$)']),
    muon_norm_jb_target_glbavgmomentum=   (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalavg_momentum'),       [r'muon norm-match ($\text{rms}\rightarrow	ext{rms}$, good norm)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \text{AVG}_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}$)']),

    muon_norm_rms_target_glbmaxmomentum=  (functools.partial(Muon, norm_kind='rms', target_norm='globalmax_momentum'),          [r'muon norm-match (rms, as if flattened as a vector)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_\text{rms}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_\text{rms}$)']),
    muon_norm_fro_target_glbmaxmomentum=  (functools.partial(Muon, norm_kind='fro', target_norm='globalmax_momentum'),          [r'muon norm-match (frobenius)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_F$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$)']),
    muon_norm_spec_target_glbmaxmomentum= (functools.partial(Muon, norm_kind='spectral', target_norm='globalmax_momentum'),     [r'muon norm-match (spectral)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_2$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_2$)']),
    muon_norm_jb_target_glbmaxmomentum=   (functools.partial(Muon, norm_kind='jbnorm', target_norm='globalmax_momentum'),       [r'muon norm-match ($\text{rms}\rightarrow	ext{rms}$, good norm)',
                                                                                                                                 r'(normalize each $||\Delta W_i||_{\text{rms}\rightarrow\text{rms}}$ to match modula-inspired $||\text{momentum}||_M$,',
                                                                                                                                 r'where $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_{\text{rms}\rightarrow\text{rms}}$)']),

    # muon_renorm_fro=                (functools.partial(Muon, renormalize='momentum', renorm_kind='fro'),                               [r'muon renorm',
    #                                                                                                                                     r'($||\Delta W||_F$ match $||\text{momentum}||_F$)']),
    # muon_renorm_spec=               (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral'),                          [r'muon renorm',
    #                                                                                                                                     r'($||\Delta W||_2$ match $||\text{momentum}||_2$)']),

    # muon_renorm_glbsfro=            (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='fro'),                     [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbmfro=            (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='fro'),                     [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbsspec=           (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='spectral'),                [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_renorm_glbmspec=           (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='spectral'),                [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)']),

    # muon_noscale=                   (functools.partial(Muon, scale=None),                                                              [r'muon no scale']),
    # muon_jbscale=                   (functools.partial(Muon, scale='jxbz'),                                                            [r'muon scale to unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$*$||\Delta W||_2$',
    #                                                                                                                                     r'(default is unit $||\Delta W||_\text{RMS}$)']),

    # muon_renorm_fro_noscale=        (functools.partial(Muon, renormalize='momentum', renorm_kind='fro', scale=None),                   [r'muon renorm',
    #                                                                                                                                     r'($||\Delta W||_F$ match $||\text{momentum}||$)',
    #                                                                                                                                     r'+ no scale']),

    # muon_renorm_spec_noscale=       (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral', scale=None),              [r'muon renorm',
    #                                                                                                                                     r'($||\Delta W||_2$ match $||\text{momentum}||$)',
    #                                                                                                                                     r'+ no scale']),

    # muon_renorm_fro_jbscale=        (functools.partial(Muon, renormalize='momentum', renorm_kind='fro', scale='jxbz'),                 [r'muon renorm',
    #                                                                                                                                     r'($||\Delta W||_F$ match $||\text{momentum}||$)',
    #                                                                                                                                     r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_spec_jbscale=       (functools.partial(Muon, renormalize='momentum', renorm_kind='spectral', scale='jxbz'),            [r'muon renorm',
    #                                                                                                                                     r'($||\Delta W||_2$ match $||\text{momentum}||$)',
    #                                                                                                                                     r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbsfro_jbscale=    (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='fro', scale='jxbz'),       [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                     r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbmfro_jbscale=    (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='fro', scale='jxbz'),       [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_F$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                     r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbsspec_jbscale=   (functools.partial(Muon, renormalize='globalsum_momentum', renorm_kind='spectral', scale='jxbz'),  [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \sum_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                     r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),

    # muon_renorm_glbmspec_jbscale=   (functools.partial(Muon, renormalize='globalmax_momentum', renorm_kind='spectral', scale='jxbz'),  [r'muon renorm',
    #                                                                                                                                     r'(modula-inspired $||\mathcal{W}||_M := \max_i\ s_i\ ||W_i||_2$,',
    #                                                                                                                                     r'$||\Delta W||_M$ match $||\text{momentum}||_M$)',
    #                                                                                                                                     r'+ unit $\sqrt{\frac{\text{fanout}}{\text{fanin}}}$ scale']),
)

if __name__ == '__main__':
    import sys
    # argv: [optim, lr, seed]
    optim_kind, lr, seed = sys.argv[1:]
    lr = float(lr)
    seed = int(seed)
    print(optim_kind, lr, seed)

    torch.manual_seed(seed + 21436)
    torch.cuda.manual_seed(seed + 21436)
    np.random.seed(seed + 21436)
    random.seed(seed + 21436)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    file = f'241018_300steps_bzs2048/orth_{optim_kind}_lr{lr:g}_seed{seed}.pth'
    if os.path.exists(file):
        print(f'skipping {file}')
        sys.exit()

    with open(file + '.running', 'wb') as f:
        f.write(b'')

    print(f'training {file}')
    model = make_model()
    model.load_state_dict(torch.load('orth_init_weights.pth', weights_only=False)[seed])
    res = train_mnist(model, opt=OPTIM_MAP[optim_kind][0], w_save_key=None, lr=lr, nsteps=500, log_nsteps=5, batch_size=2048)
    torch.save(res._asdict(), file)
    print(f'saved {file}')
