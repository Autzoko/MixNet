import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from packages.mamba.mamba_ssm import Mamba as MambaSSM
    MAMBA_AVAIL = True
except ImportError:
    print('Warning: maba-ssm not installed.')
    MAMBA_AVAIL = False


class OptimizedMambaBlock(nn.Module):
    def __init__(self, dim, state_dim=16, use_mamba=True):
        super().__init__()
        self.dim = dim
        self.use_mamba = use_mamba and MAMBA_AVAIL

        self.norm = nn.LayerNorm(dim)
        if self.use_mamba:
            self.mamba = MambaSSM(
                d_model=dim,
                d_state=state_dim,
                d_conv=4,
                expand=2,
                bimamba_type=None
            )
        else:
            self.fallback_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU()
            )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm_mlp = nn.LayerNorm(dim)


    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        identity = x

        if self.use_mamba:
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = x_flat + self.mamba(self.norm(x_flat))
            x_flat = x_flat + self.mlp(self.norm_mlp(x_flat))
            x = x_flat.transpose(1, 2).view(B, C, H, W)
        else:
            x = x + self.fallback_conv(x)
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = x_flat + self.mlp(self.norm_mlp(x_flat))
            x = x_flat.transpose(1, 2).view(B, C, H, W)

        return x



class SimplifiedSSM(nn.Module):
    """
    Simplified State Space Model (SSM) layer for sequence modeling.
    """
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        self.input_proj = nn.Linear(dim, dim * 2)
        self.output_proj = nn.Linear(dim, dim)

        self.A_log = nn.Parameter(torch.randn(state_dim))
        self.D = nn.Parameter(torch.randn(dim))

        self.x_proj = nn.Linear(dim, state_dim * 2)

        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        """
        x: (B, L, D): batch, sequence length, dimension
        """
        B, L, D = x.shape

        x_proj = self.input_proj(x)  # (B, L, 2D)
        x_gate, x_input = x_proj.chunk(2, dim=-1)  # (B, L, D), (B, L, D)
        x_gate = F.silu(x_gate)

        x_conv = self.conv1d(x_input.transpose(1, 2)).transpose(1, 2)  # (B, L, D)

        BC = self.x_proj(x_input)
        B_sel, C_sel = BC.chunk(2, dim=-1)  # (B, L, state_dim), (B, L, state_dim)

        A = -torch.exp(self.A_log)  # (state_dim,)
        state = torch.zeros(B, self.state_dim, D, device=x.device)
        outputs = []

        for i in range(L):
            b = B_sel[:, i: i + 1, :]  # (B, 1, state_dim)
            c = C_sel[:, i: i + 1, :]  # (B, 1, state_dim)
            u = x_conv[:, i: i + 1, :]  # (B, 1, D)

            state = A.view(1, -1, 1) * state + b.transpose(1, 2) * u
            y = (c.transpose(1, 2) * state).sum(dim=1, keepdim=True)  # (B, 1, D)
            outputs.append(y)

        y = torch.cat(outputs, dim=1)  # (B, L, D)

        y = y + self.D * x_input
        y = y * x_gate
        y = self.output_proj(y)

        return y
    

class MambaBlock(OptimizedMambaBlock):
    """
    Mamba Block based on SSM, support bi-directional context.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """
    pass