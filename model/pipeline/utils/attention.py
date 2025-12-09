import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self Attention Module.
    """

    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (num_windows*B, window_size*window_size, C)
        """

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x
    

def window_partition(x, window_size):
    """
    Partition input into windows.
    x: (B, H, W, C)  <- 注意：输入格式是 (B, H, W, C)，不是 (B, C, H, W)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # 修复：按照正确的维度顺序进行 reshape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse windows back to original input.
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, H, W, C)  <- 注意：输出格式应该是 (B, H, W, C)，与输入一致
    """
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 修复：reshape 和 permute 的顺序，使输出格式为 (B, H, W, C)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


class WindowAttentionBlock(nn.Module):
    """
    Window-based Multi-Head Self Attention Block.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        B, C, H, W = x.shape
        identity = x

        # Padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        # 转换为 (B, H, W, C) 格式
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        # Window partition
        x_windows = window_partition(x, self.window_size)  # (num_windows*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (num_windows*B, window_size*window_size, C)

        # Attention
        attn_windows = self.attn(self.norm1(x_windows))  # (num_windows*B, window_size*window_size, C)
        attn_windows = attn_windows + x_windows

        # MLP
        mlp_windows = self.mlp(self.norm2(attn_windows))  # (num_windows*B, window_size*window_size, C)
        mlp_windows = mlp_windows + attn_windows

        # Reverse windows
        mlp_windows = mlp_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(mlp_windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)

        # 转换回 (B, C, H, W) 格式
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        
        return x