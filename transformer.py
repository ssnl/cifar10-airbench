# type: ignore

# from jacob

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Identity(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)


class Bottle1D(nn.Module):
    inner: nn.Sequential

    def __init__(self, *modules) -> None:
        super().__init__()
        self.inner = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)



class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.gelu(x2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)



class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)





def make_activation_fn(name: Optional[str]) -> Tuple[nn.Module, int]:
    # returns (activation_fn, size reduction factor)
    if name is None:
        return nn.Identity(), 1
    if name == 'mish':
        return nn.Mish(), 1
    elif name == 'relu':
        return nn.ReLU(), 1
    elif name == 'geglu':
        return GEGLU(), 2
    elif name == 'swiglu':
        return SwiGLU(), 2
    elif name == 'gelu':
        return nn.GELU(), 1
    else:
        raise ValueError(f"Unknown activation function: {name}")


class MLP(torch.nn.Module):
    def __init__(self, depth, width, input_dim, output_dim, act: str = 'gelu', init_scale=1.):
        super().__init__()
        assert depth == 2
        self.width = width
        self.act, act_reduction_factor = make_activation_fn(act)
        self.fc1 = nn.Linear(input_dim, width * act_reduction_factor, bias=False)
        self.fc2 = nn.Linear(width, output_dim, bias=False)
        self.init_weights(init_scale)

    def init_weights(self, init_scale=1):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        with torch.no_grad():
            self.fc2.weight.mul_(init_scale)
        return

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadSelfAttention(nn.Module):
    """ MultiHead Attention using PyTorch's scaled_dot_product_attention """

    def __init__(self, n_embd, n_head, causal=False, bias=False, init_scale=1.):
        super().__init__()
        self.causal = causal
        self.heads = n_head
        self.in_proj = nn.Linear(n_embd, n_embd * 3, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.init_weights(init_scale)

    def init_weights(self, init_scale=1):
        """
        Using same initialization protocol for PyTorch's MultiheadAttention
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L1041
        """
        torch.nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            torch.nn.init.constant_(self.in_proj.bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)
        with torch.no_grad():
            self.out_proj.weight.mul_(init_scale)
        return

    def in_projection(self, x):
        """
        Args:
            q, k, v: torch.Tensor of shape (B, S, D)
        Returns:
            q, k, v: torch.Tensor of shape (B, H, S, D_head)
        """
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k, v = (
            q.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
            k.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
            v.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
        )
        return q, k, v

    def forward(self, x):
        q, k, v = self.in_projection(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.permute(0, 2, 1, 3).flatten(-2, -1)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return f"heads={self.heads}, causal={self.causal}"


class Block(nn.Module):

    def __init__(self, n_head, n_embd, causal, mlp_ratio=4, mlp_method='default', act='gelu', residual_init_scale=1.):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, causal, bias=False, init_scale=residual_init_scale)
        self.ln_2 = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.mlp = MLP(depth=2, width=mlp_ratio*n_embd, input_dim=n_embd, output_dim=n_embd, act=act, init_scale=residual_init_scale)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, n_head, n_embd, n_layer, block_size, vocab_size, mlp_method='default'):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wt_embedding = nn.Embedding(vocab_size, n_embd),
            wp_embedding = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([
                Block(n_head, n_embd, causal=True, mlp_method=mlp_method) for _ in range(n_layer)
                ]),
            ln_f = nn.LayerNorm(n_embd, elementwise_affine=True),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.transformer.wt_embedding.weight = self.lm_head.weight


    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wt_embedding(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wp_embedding(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        return self.lm_head(x)


class ViT(nn.Module):

    def __init__(self, n_head, n_embd, n_layer, num_patches, img_size, num_channels, num_classes):
        super().__init__()

        self.block_size = num_patches**2 + 1
        self.patch_size = img_size // num_patches
        pixels_per_patch = self.patch_size**2 * num_channels

        self.transformer = nn.ModuleDict(dict(
            patch_embedding = nn.Linear(pixels_per_patch, n_embd),
            position_embedding = nn.Embedding(self.block_size, n_embd),
            h = nn.ModuleList([Block(n_head, n_embd, self.block_size, causal=False) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, elementwise_affine=True),
        ))
        self.cls_token = nn.Parameter(torch.randn(n_embd))
        self.lm_head = nn.Linear(n_embd, num_classes)


    def forward(self, x):
        n_patch = x.shape[2] // self.patch_size * x.shape[3] // self.patch_size
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        patches = patches.reshape(x.size(0), n_patch, -1)

        pos = torch.arange(0, self.block_size, dtype=torch.long, device=patches.device)
        pos_emb   = self.transformer.position_embedding(pos)
        patch_emb = self.transformer.patch_embedding(patches)
        patch_emb = torch.cat([self.cls_token.view(1,1,self.cls_token.shape[0]).repeat(patch_emb.size(0),1,1), patch_emb],dim=1)

        x = patch_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x[:,0])
