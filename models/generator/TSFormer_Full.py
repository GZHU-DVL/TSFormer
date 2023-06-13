## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
## Encoder
# def ones(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0.5)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Spatial_Window_Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, input_resolution, drop_path, shift_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)).cuda()  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qkv_edge = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=qkv_bias)

        self.qkv_edge = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv_edge_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # fully connected layer in Fig.2
        # self.fc = nn.Conv2d(3 * self.num_heads, 9, kernel_size=1, bias=True)
        # group convolution layer in Fig.3
        # self.dep_conv = nn.Conv2d(9 * dim // self.num_heads, dim, kernel_size=3, bias=True,
        #                           groups=dim // self.num_heads, padding=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # add alpha
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # rates for both paths
        # self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        # self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # ones(self.rate1)
    #     # ones(self.rate2)
    #     # shift initialization for group convolution
    #     kernel = torch.zeros(9, 3, 3)
    #     for i in range(9):
    #         kernel[i, i // 3, i % 3] = 1.
    #     kernel = kernel.squeeze(0).repeat(self.dim, 1, 1, 1)
    #     self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
    #     self.dep_conv.bias = zeros(self.dep_conv.bias)

    def forward(self, x, edge, mask=None):
        """
        Args:
            x: input features with shape of (B, H, W, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv_edge = self.qkv_edge_dwconv(self.qkv_edge(edge))

        qkv = rearrange(qkv, 'b c h w -> b h w c')
        qkv_edge = rearrange(qkv_edge, 'b c h w -> b h w c')

        # update
        if self.shift_size > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            qkv_edge = torch.roll(qkv_edge, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # qkv = self.qkv(x)
        # qkv_edge = self.qkv_edge(edge)

        H, W = self.input_resolution

        # fully connected layer
        # f_all = qkv.reshape(x.shape[0], H*W, 3*self.num_heads, -1).permute(0, 2, 1, 3) # B, 3*nhead, H*W, C//nhead
        # f_conv = self.fc(f_all).permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[-1]//self.num_heads, H, W) # B, 9*C//nhead, H, W

        # group conovlution
        # out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 1) # B, H, W, C

        # partition windows
        qkv = window_partition(qkv, self.window_size[0])  # nW*B, window_size, window_size, C
        qkv_edge = window_partition(qkv_edge, self.window_size[0])

        B_, _, _, C = qkv.shape

        qkv = qkv.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C
        qkv_edge = qkv_edge.view(-1, self.window_size[0] * self.window_size[1], C)

        N = self.window_size[0] * self.window_size[1]
        C = C // 3

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_edge = qkv_edge.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = qkv_edge[1], qkv_edge[2]

        k = k_e * self.alpha1 + k
        v = v_e * self.alpha2 + v

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size[0], H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = self.drop_path(x)

        x = rearrange(x, 'b h w c -> b c h w')

        return x


class FeedForwardEncoder(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForwardEncoder, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class AttentionEncoder(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(AttentionEncoder, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_edge = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.qkv_edge = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_edge = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                         bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out_edge = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # add alpha
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        # self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        # self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, inp_img, inp_edge):
        b, c, h, w = inp_img.shape

        qkv = self.qkv_dwconv(self.qkv(inp_img))

        qkv_edge = self.qkv_dwconv_edge(self.qkv_edge(inp_edge))

        q, k, v = qkv.chunk(3, dim=1)

        q_e, k_e, v_e = qkv_edge.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_e = rearrange(q_e, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_e = rearrange(k_e, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_e = rearrange(v_e, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q_e = torch.nn.functional.normalize(q_e, dim=-1)
        k_e = torch.nn.functional.normalize(k_e, dim=-1)

        # update
        k = k_e * self.alpha1 + k
        v = v_e * self.alpha2 + v

        # k_t = torch.cat([k, k_e], dim=2)
        # v_t = torch.cat([v, v_e], dim=2)

        attn_edge = (q_e @ k_e.transpose(-2, -1)) * self.temperature_edge  # (C/2, C/2)
        attn_edge = attn_edge.softmax(dim=-1)

        out_edge = (attn_edge @ v_e)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (C/2, C/2)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # (C/2, C/2)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_edge = rearrange(out_edge, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out_edge = self.project_out_edge(out_edge)
        return out, out_edge


##########################################################################
class TransformerBlockEncoder(nn.Module):
    def __init__(self, dim, num_heads, spatial_head, input_resolution, window_size, shift_size, ffn_expansion_factor, drop_path, bias, LayerNorm_type):
        super(TransformerBlockEncoder, self).__init__()

        self.norm1_channel = LayerNorm(dim // 2, LayerNorm_type)
        self.norm1_spatial = LayerNorm(dim // 2, LayerNorm_type)
        self.norm1_edge = LayerNorm(dim // 2, LayerNorm_type)
        self.attn = AttentionEncoder(dim // 2, num_heads, bias)
        self.norm2 = LayerNorm(dim + dim // 2, LayerNorm_type)
        self.ffn = FeedForwardEncoder(dim + dim // 2, ffn_expansion_factor, bias)
        self.dim = dim

        self.input_resolution = input_resolution

        if shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            img_mask = img_mask.to(torch.device('cuda'))
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.spatialAttn = Spatial_Window_Attention(dim=dim // 2, window_size=to_2tuple(window_size), num_heads=spatial_head, input_resolution=input_resolution, drop_path=drop_path, shift_size=shift_size)

    def forward(self, x):
        inp_img, inp_edge = x
        inp_img_channel, inp_img_spatial = torch.split(inp_img, [self.dim // 2, self.dim // 2], dim=1)

        origin_img_channel = inp_img_channel
        origin_img_spatial = inp_img_spatial
        origin_edge = inp_edge

        inp_img_channel = self.norm1_channel(inp_img_channel)
        inp_img_spatial = self.norm1_spatial(inp_img_spatial)
        inp_edge = self.norm1_edge(inp_edge)

        attn_img, attn_edge = self.attn(inp_img_channel, inp_edge)
        attn_spatial = self.spatialAttn(inp_img_spatial, inp_edge, mask=self.attn_mask)

        inp_img_channel = origin_img_channel + attn_img
        inp_edge = origin_edge + attn_edge
        inp_img_spatial = origin_img_spatial + attn_spatial

        inp_cat = torch.cat((inp_edge, inp_img_channel, inp_img_spatial), dim=1)
        inp_cat = inp_cat + self.ffn(self.norm2(inp_cat))
        inp_edge, inp_img = torch.split(inp_cat, [self.dim // 2, self.dim], dim=1)
        return inp_img, inp_edge


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=24, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim - 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, inp_hog):
        x = self.proj(x)
        return torch.cat((x, inp_hog), dim=1)


class OverlapPatchEmbedEdge(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbedEdge, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, LayerNorm_type='WithBias'):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = LayerNorm(planes, LayerNorm_type)
        self.conv2 = nn.Conv2d(planes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = LayerNorm(inplanes, LayerNorm_type)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


class Feature2Structure(nn.Module):

    def __init__(self, inplanes=64, planes=16):
        super(Feature2Structure, self).__init__()

        self.structure_resolver = Bottleneck(inplanes, planes)
        self.out_layer = nn.Sequential(
            nn.Conv2d(inplanes, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, structure_feature):
        x = self.structure_resolver(structure_feature)
        structure = self.out_layer(x)
        return structure


class StructureEnhancement(nn.Module):
    def __init__(self,
                 edge_channels=3,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=edge_channels, out_channels=64, kernel_size=7, padding=0)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(256), num_heads=8, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(8)])

        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.convt2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.convt3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0)

        self.act_last = nn.Sigmoid()

    def forward(self, img):
        x = self.pad1(img)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.act(x)

        x = self.latent(x)

        x = self.convt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)

        edge, hog = torch.split(x, [1, 1], dim=1)

        return self.act_last(edge), hog


##########################################################################
##---------- TSFormer -----------------------
class TSFormer_Full(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 # edge_channels=2,
                 dim=24,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 spatial_heads=[1, 3, 6, 12],
                 window_size=8,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(TSFormer_Full, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.edge_generate = StructureEnhancement()

        self.patch_embed_edge = OverlapPatchEmbedEdge(1, dim // 2)

        dpr = [x.item() for x in torch.linspace(0, 0.3, sum(num_blocks))]

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlockEncoder(dim=dim, num_heads=heads[0], spatial_head=spatial_heads[0], input_resolution=(256, 256),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:1])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_2_edge = Downsample(dim // 2)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlockEncoder(dim=int(dim * 2 ** 1), num_heads=heads[1], spatial_head=spatial_heads[1], input_resolution=(128, 128),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:1]):sum(num_blocks[:2])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_3_edge = Downsample(int(dim))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlockEncoder(dim=int(dim * 2 ** 2), num_heads=heads[2], spatial_head=spatial_heads[2], input_resolution=(64, 64),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:2]):sum(num_blocks[:3])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.down3_4_edge = Downsample(int(dim * 2 ** 1))
        self.latent = nn.Sequential(*[
            TransformerBlockEncoder(dim=int(dim * 2 ** 3), num_heads=heads[3], spatial_head=spatial_heads[3], input_resolution=(32, 32),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:3]):sum(num_blocks[:4])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.up4_3_edge = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level3_edge = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlockEncoder(dim=int(dim * 2 ** 2), num_heads=heads[2], spatial_head=spatial_heads[2],
                                    input_resolution=(64, 64),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:2]):sum(num_blocks[:3])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_2_edge = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_edge = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlockEncoder(dim=int(dim * 2 ** 1), num_heads=heads[1], spatial_head=spatial_heads[1], input_resolution=(128, 128),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:1]):sum(num_blocks[:2])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_edge = Upsample(int(dim * 1 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlockEncoder(dim=dim * 2 ** 1, num_heads=heads[0], spatial_head=spatial_heads[0], input_resolution=(256, 256),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:1])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlockEncoder(dim=dim * 2 ** 1, num_heads=heads[0], spatial_head=spatial_heads[0],
                                    input_resolution=(256, 256),
                                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    drop_path=dpr[sum(num_blocks[:0]):sum(num_blocks[:1])][i], bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_edge = nn.Conv2d(int(dim * 1 ** 1), 1, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act_last = nn.Sigmoid()

    def forward(self, inp_img, inp_edge, inp_hog):

        first_out_edge, first_out_hog = self.edge_generate(torch.cat((inp_edge, inp_hog), dim=1))

        inp_enc_level1 = self.patch_embed(inp_img, first_out_hog)

        inp_enc_level1_edge = self.patch_embed_edge(first_out_edge)

        out_enc_level1, out_enc_level1_edge = self.encoder_level1((inp_enc_level1, inp_enc_level1_edge))

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2_edge = self.down1_2_edge(out_enc_level1_edge)

        out_enc_level2, out_enc_level2_edge = self.encoder_level2((inp_enc_level2, inp_enc_level2_edge))

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc_level3_edge = self.down2_3_edge(out_enc_level2_edge)

        out_enc_level3, out_enc_level3_edge = self.encoder_level3((inp_enc_level3, inp_enc_level3_edge))

        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4_edge = self.down3_4_edge(out_enc_level3_edge)
        latent, latent_out = self.latent((inp_enc_level4, inp_enc_level4_edge))

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3_edge = self.up4_3_edge(latent_out)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3_edge = torch.cat([inp_dec_level3_edge, out_enc_level3_edge], 1)
        inp_dec_level3_edge = self.reduce_chan_level3_edge(inp_dec_level3_edge)
        out_dec_level3, out_dec_level3_edge = self.decoder_level3((inp_dec_level3, inp_dec_level3_edge))

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2_edge = self.up3_2_edge(out_dec_level3_edge)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2_edge = torch.cat([inp_dec_level2_edge, out_enc_level2_edge], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2_edge = self.reduce_chan_level2_edge(inp_dec_level2_edge)
        out_dec_level2, out_dec_level2_edge = self.decoder_level2((inp_dec_level2, inp_dec_level2_edge))

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1_edge = self.up2_1_edge(out_dec_level2_edge)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1_edge = torch.cat([inp_dec_level1_edge, out_enc_level1_edge], 1)
        out_dec_level1, out_dec_level1_edge = self.decoder_level1((inp_dec_level1, inp_dec_level1_edge))

        out_dec_level1, out_dec_level1_edge = self.refinement((out_dec_level1, out_dec_level1_edge))

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
            out_dec_level1_edge = self.output_edge(out_dec_level1_edge)
            out_edge = self.act_last(out_dec_level1_edge)

        return out_dec_level1, out_edge, first_out_edge, first_out_hog
