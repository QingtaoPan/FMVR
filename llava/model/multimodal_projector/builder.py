import torch
import torch.nn as nn
import re

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"projector type: {projector_type}")
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.mm_hidden_size*16)]
        # modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.mm_hidden_size*16, config.hidden_size))
        # modules.append(nn.GELU())
        # modules.append(nn.Linear(config.mm_hidden_size*16, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_projector_crosslayer(config, delay_load=False, **kwargs):

    # return nn.Linear(config.mm_hidden_size*2, config.mm_hidden_size)

    modules = [nn.Linear(config.mm_hidden_size*8, config.mm_hidden_size*4)]
    modules.append(nn.GELU())
    modules.append(nn.Linear(config.mm_hidden_size*4, config.mm_hidden_size))

    return nn.Sequential(*modules)

def build_vision_projector_baseline(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"projector type teachers: {projector_type}")
    if projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class AvgBranch(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.low_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out + x


class LocalAvgBranch(nn.Module):
    def __init__(self, dim) -> None:
        super(LocalAvgBranch, self).__init__()

        self.gap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.low_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out + x


class StripGlobalAvgBranch(nn.Module):
    def __init__(self, dim, size) -> None:
        super(StripGlobalAvgBranch, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(size)

        self.low_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        low_frequency = self.gap(x)
        high_frequency = x - low_frequency
        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + x * self.b + out
        return out


class StripGlobalMaxBranch(nn.Module):
    def __init__(self, dim, kernel) -> None:
        super().__init__()

        self.mp = nn.AdaptiveMaxPool2d((kernel))

        self.low_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        high_frequency = self.mp(x)
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out


class MaxBranch(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.low_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        high_frequency = self.mp(x)
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out


class MaxDilateBranch(nn.Module):
    def __init__(self, dim) -> None:
        super(MaxDilateBranch, self).__init__()

        dilation = 2
        kernel = 5

        self.pad = nn.ReflectionPad2d(dilation * (kernel - 1) // 2)

        self.mp = nn.MaxPool2d(kernel_size=kernel, dilation=dilation, stride=1)

        self.low_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.high_weight = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)

        self.a = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        high_frequency = self.mp(self.pad(x))
        low_frequency = x - high_frequency

        out = low_frequency * self.low_weight + high_frequency * (1. + self.high_weight)
        out = x * high_frequency * self.a + self.b * x + out

        return out


class AvgMax(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gap = AvgBranch(dim)
        self.mp = MaxBranch(dim)
        self.local_gap = LocalAvgBranch(dim)
        self.globa_horizontal_avg = StripGlobalAvgBranch(dim, (None, 1))
        self.globa_vertical_avg = StripGlobalAvgBranch(dim, (1, None))
        self.global_horizontal_max = StripGlobalMaxBranch(dim, (None, 1))
        self.global_vertial_max = StripGlobalMaxBranch(dim, (1, None))
        # self.maxdilation = MaxDilateBranch(dim)

    def forward(self, x):
        x1 = self.gap(x)
        x2 = self.mp(x)
        # x3 = self.local_gap(x)
        # x4 = self.globa_horizontal_avg(x)
        # x5 = self.globa_vertical_avg(x)
        # x6 = self.global_horizontal_max(x)
        # x7 = self.global_vertial_max(x)
        # x8 = self.maxdilation(x)
        return x1 + x2

class Decoder(nn.Module):
    def __init__(self, d=4096, n_out=576):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, n_out, d))

    def forward(self, V):
        q = self.query.expand(V.size(0), -1, -1)  # [B, 576, d]
        out, _ = self.attn(q, V, V)
        return self.proj(out)

