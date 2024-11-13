from typing import *
import torch
import torch.nn as nn
import functools
import numpy as np
import os
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def zeropower_via_svd(G, steps=None, dtype=torch.float32, **kwargs):
    U, S, V = G.to(dtype).svd()
    return (U @ V.T).to(G.dtype)

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7, dtype=torch.float32,
                                 abc: torch.Tensor = torch.tensor((3.4445, -4.7750,  2.0315)),
                                 normalize=True):
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
    X = G.to(dtype)
    if normalize:
        X = X / (X.norm() + eps)
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
    newtonschulz5_nonorm=functools.partial(zeropower_via_newtonschulz5, normalize=False),
    newtonschulz5_proper=functools.partial(zeropower_via_newtonschulz5, abc=torch.tensor((1.5, -0.5, 0))),
    newtonschulz5_sched5=functools.partial(zeropower_via_newtonschulz5, steps=8, abc=make_schedule(3, 1, 1)),
    newtonschulz5_sched8=functools.partial(zeropower_via_newtonschulz5, steps=8, abc=make_schedule(7, 1, 0)),
    newtonschulz5_sched10=functools.partial(zeropower_via_newtonschulz5, steps=10, abc=make_schedule(8, 1, 1)),
    newtonschulz5_sched14=functools.partial(zeropower_via_newtonschulz5, steps=14, abc=make_schedule(10, 2, 2)),
)

def procedure(W, G, lr=1e-2, backend='svd', norm='analytical'):
    C = W.T @ G
    C_asym = (C - C.T) / 2
    X = zeropower_backends[backend](C_asym)
    A = W @ X
    # print((W.T @ A + A.T @ W).norm(), A.svd().S[0], W.svd().S[0], X.svd().S[0])
    Ap = A * (-lr)
    Wn = W + Ap
    # print((Wn @ Wn.T) / (1 + lr ** 2))
    if norm == 'analytical':
        Wn = Wn / ((1 + lr ** 2) ** 0.5)
    elif norm == 'empirical':
        Wn = Wn * ( min(Wn.shape) **0.5 / (Wn.norm() + 1e-6))
    else:
        raise ValueError(f'Unknown norm: {norm}')
    return Wn

class OrthOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, backend='svd', norm='analytical', reproj_interval=50, reproj_fn=None,
                 nesterov=True, momentum=0.9):
        defaults = dict(lr=lr, backend=backend, norm=norm, reproj_interval=reproj_interval, reproj_fn=reproj_fn,
                        nesterov=nesterov, momentum=momentum)
        super().__init__(params, defaults)

    def _apply_momentum(self, state, x: torch.Tensor, momentum, *, is_nesterov, prefix: str):
        # m = beta1 * m + g
        # g = g + m * beta1
        # ----
        # equiv:
        # g = g + beta1**2 * m + beta1 * g
        #   = (1+beta1) g + beta1**2 m
        if momentum == 0:
            return x
        if f'{prefix}_momentum_buffer' not in state:
            state[f'{prefix}_momentum_buffer'] = torch.zeros_like(x)
        buf = state[f'{prefix}_momentum_buffer']
        if is_nesterov:
            buf.mul_(momentum).add_(x)
            return x.add(buf, alpha=momentum)
        else:
            torch.lerp(buf, x, 1 - momentum, out=buf)
            return buf / (1 - momentum ** state['stept'])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]

                if 'stept' not in state:
                    state['stept'] = 0
                state['stept'] += 1
                stept = state['stept']

                update = self._apply_momentum(
                    state, grad, group['momentum'], is_nesterov=group['nesterov'], prefix='update')

                scale_factor = (p.data.shape[0] / p.data.shape[1]) ** 0.5

                p.data.copy_(
                    procedure(p.data / scale_factor,
                              update,
                            #   update * scale_factor,  scaling doesn't matter for this update rule
                              lr=group['lr'], backend=group['backend'], norm=group['norm'])
                    * scale_factor
                )

                if stept % group['reproj_interval'] == 0 and (reproj_fn := group['reproj_fn']) is not None:
                    p.data.copy_(reproj_fn(p.data / scale_factor) * scale_factor)

#################
# Model
#################

class Mul(nn.Module):
    def __init__(self, scale, learnable=False):
        super().__init__()
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

def make_model(add_relu_scale=True, final_scale_fn=nn.Identity):
    model = nn.Sequential(
        nn.Linear(28*28, 28*28),
        nn.ReLU(),
        Mul(np.sqrt(2)) if add_relu_scale else nn.Identity(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        Mul(np.sqrt(2)) if add_relu_scale else nn.Identity(),
        nn.Linear(256, 256),
        nn.ReLU(),
        Mul(np.sqrt(2)) if add_relu_scale else nn.Identity(),
        nn.Linear(256, 256),
        nn.ReLU(),
        Mul(np.sqrt(2)) if add_relu_scale else nn.Identity(),
        nn.Linear(256, 64),
        nn.ReLU(),
        Mul(np.sqrt(2)) if add_relu_scale else nn.Identity(),
        nn.Linear(64, 10),
        final_scale_fn(),
    ).to(torch.float32)

    model = orth_init(model).to(device)
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

def make_optimizers(model, lr, **kwargs):
    return [
        OrthOptimizer([p for p in model.parameters() if p.ndim == 2], lr=lr, **kwargs),
        torch.optim.Adam([p for p in model.parameters() if p.ndim != 2], lr=lr),
    ]

#################
# Training
#################

# mnist training loop

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=2048, shuffle=True, drop_last=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=4096, shuffle=False)


def eval_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device).flatten(1), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(loader.dataset)


def train_model(model, optims, train_loader, test_loader, epochs=20):
    test_accs = []
    sds = []
    losses = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).flatten(1), target.to(device)
            [optim.zero_grad() for optim in optims]
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            [optim.step() for optim in optims]
            losses.append(loss.item())
            if batch_idx % 20 == 0:
                print(f'epoch {epoch:02d} batch {batch_idx:03d} loss: {loss.item():.4f}')
        test_accs.append(eval_model(model, test_loader))
        sds.append({k: v.cpu() for k, v in model.state_dict().items()})

    final_train_acc, final_test_acc = eval_model(model, train_loader), test_accs[-1]
    print(f'final train acc: {final_train_acc:.2%} test acc: {final_test_acc:.2%}')
    return final_train_acc, torch.as_tensor(test_accs), torch.as_tensor(losses).reshape(epochs, -1), sds



if __name__ == '__main__':
    import sys
    # argv: [lr, scale seed]
    lr, scale, seed = sys.argv[1:]
    lr = float(lr)
    scale = float(scale)
    seed = int(seed)
    print(lr, scale, seed)

    file = f'241113_test_jb_proj_g/mnist_lr{lr:g}_scale{scale:g}_seed{seed}.pth'

    def run():
        torch.manual_seed(seed + 21436)
        torch.cuda.manual_seed(seed + 21436)
        np.random.seed(seed + 21436)
        random.seed(seed + 21436)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f'training {file}')
        model = make_model(add_relu_scale=True, final_scale_fn=lambda: Mul(scale))
        optims = make_optimizers(model, lr, backend='newtonschulz5', norm='analytical', reproj_interval=10,
                             nesterov=False, reproj_fn=None,
                            #  reproj_fn=lambda w: (
                            #      zeropower_backends['newtonschulz5_proper'](w, normalize=False, steps=1)
                            #  ),
                         )
        final_train_acc, test_accs, losses, sds = train_model(model, optims, train_loader, test_loader, epochs=25)
        torch.save(dict(final_train_acc=final_train_acc, test_accs=test_accs, losses=losses, sds=sds), file)
        print(f'saved {file}')

    with open(file + '.running', 'w') as f:
        run()
        f.write('0')
