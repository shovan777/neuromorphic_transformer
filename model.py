import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding
from config import TIME_STEP

class Img2Spike(nn.Module):
    def __init__(self, in_channels, channels, use_poisson=False):
        super().__init__()
        self.use_poisson = use_poisson
        if self.use_poisson:
            self.poisson_encoder = encoding.PoissonEncoder()
        c1 = max(1, channels // 4)
        c2 = max(1, channels // 2)
        self.spike_conv = nn.Sequential(
            layer.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c1),
            neuron.IFNode(step_mode="m", surrogate_function=surrogate.ATan()),
            layer.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c2),
            neuron.IFNode(step_mode="m", surrogate_function=surrogate.ATan()),
            layer.Conv2d(c2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(step_mode="m", surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, step_mode="m")
        
    def forward(self, x):
        if self.use_poisson:
            x = x.clamp(0.0, 1.0)
            x_seq = torch.stack(
                [self.poisson_encoder(x) for _ in range(TIME_STEP)],
                dim=0,
            )  # [T, B, C, H, W]
        else:
            x_seq = x.unsqueeze(0).repeat(TIME_STEP, 1, 1, 1, 1)  # T, B, C, H, W
        return self.spike_conv(x_seq) # [T, B, C, H, W]

class SpikePatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = layer.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.lif = neuron.IFNode(step_mode="m", surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(3).transpose(2, 3) 
        x = self.lif(x)
        return x

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(step_mode="m", v_threshold=0.5)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(step_mode="m", v_threshold=0.5)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(step_mode="m", v_threshold=0.5)

        self.attn_bn = nn.BatchNorm1d(dim)
        self.attn_lif = neuron.LIFNode(step_mode="m", v_threshold=0.5)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(step_mode="m", v_threshold=0.5)

    def forward(self, x):
        T,B,N,C = x.shape

        x_for_qkv = x.flatten(0, 1)  
        q_linear_out = self.q_linear(x_for_qkv)  
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Save attention weights for visualization (detached to avoid keeping compute graph)
        self.attn_weights = attn.detach()

        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_bn(x.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))

        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        bias=True,
    ):
        super().__init__()
        self.fc1 = layer.Linear(in_features, hidden_features, bias=bias)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron.LIFNode(step_mode="m", surrogate_function=surrogate.ATan(), v_threshold=0.5)
        self.fc2 = layer.Linear(hidden_features, out_features, bias=bias)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = neuron.LIFNode(step_mode="m", surrogate_function=surrogate.ATan(), v_threshold=0.5)
        functional.set_step_mode(self, "m")
        
    def forward(self, x):
        # x: [T, B, N, C]
        T, B, N, C = x.shape
        x = self.fc1(x)  # [T, B, N, hidden]
        x = self.fc1_bn(x.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, -1)
        x = self.fc1_lif(x)
        x = self.fc2(x)  # [T, B, N, out]
        x = self.fc2_bn(x.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, -1)
        x = self.fc2_lif(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x

class SpikFormer(nn.Module):
    def __init__(
        self,
        num_classes=10,
        in_channels=1,
        num_channels=1,
        use_poisson=False,
        img_size=28,
        patch_size=4,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.,
        use_cupy=False,
    ):
        super().__init__()
        self.dim = dim
        self.img2spike = Img2Spike(
            in_channels=in_channels,
            channels=num_channels,
            use_poisson=use_poisson,
        )
        self.patch_embed = SpikePatchEmbed(in_channels=num_channels, embed_dim=dim, patch_size=patch_size)
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.pos_enc = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) 
        functional.set_step_mode(self, "m")

        if use_cupy:
            functional.set_backend(self, "cupy")

    def forward(self, x):
        x = self.img2spike(x)
        x = self.patch_embed(x)
        x = x + self.pos_enc
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=0) # mean over time dimension
        x = x.mean(dim=1) # Global average pooling instead of class token
        x = self.head(x)
        return x
