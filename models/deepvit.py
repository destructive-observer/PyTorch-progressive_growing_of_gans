import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)

        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            # print('attn {}'.format(attn))
            # print('ff {}'.format(ff))
            print('attn before x {}'.format(x))
            x = attn(x)
            print('attn after x {}'.format(x))
            x = ff(x)
            print('ff after x {}'.format(x))
            t = nn.Tanh()
            x = t(x)
        return x

class DeepViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        print('img shape value{}'.format(img.shape))
        x = self.to_patch_embedding(img)
        print('x.shape shape value{}'.format(x.shape))
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        print('cls_tokens.shape shape value{}'.format(cls_tokens.shape))
        # x = torch.cat((cls_tokens, x), dim=1)
        print('x.shape shape value{}'.format(x.shape))
        x += self.pos_embedding[:, :64]
        print('x.pos_embedding shape value{}'.format(self.pos_embedding.shape))
        print('x.shape shape value{}'.format(x.shape))
        x = self.dropout(x)
        print('x.dropout shape shape value{}'.format(x.shape))
        x = self.transformer(x)
        print('x.shape11 transformer shape value{}'.format(x.shape))
        x =rearrange(x,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 32, p2 = 32,c=3,h=8,w=8)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # print('x.shape22 transformer shape value{}'.format(x.shape))
        # x = self.to_latent(x)
        # print('x.shape33 transformer shape value{}'.format(x.shape))
        # return self.mlp_head(x)
        return x
# img shape valuetorch.Size([1, 3, 256, 256])
# x.shape shape valuetorch.Size([1, 64, 1024])
# cls_tokens.shape shape valuetorch.Size([1, 1, 1024])
# x.shape shape valuetorch.Size([1, 65, 1024])
# x.pos_embedding shape valuetorch.Size([1, 65, 1024])
# x.shape shape valuetorch.Size([1, 65, 1024])
# x.dropout shape shape valuetorch.Size([1, 65, 1024])
# attn Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): Attention(
#       (to_qkv): Linear(in_features=1024, out_features=3072, bias=False)
#       (reattn_norm): Sequential(
#         (0): Rearrange('b h i j -> b i j h')
#         (1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
#         (2): Rearrange('b i j h -> b h i j')
#       )
#       (to_out): Sequential(
#         (0): Linear(in_features=1024, out_features=1024, bias=True)
#         (1): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# ff Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=1024, out_features=2048, bias=True)
#         (1): GELU()
#         (2): Dropout(p=0.1, inplace=False)
#         (3): Linear(in_features=2048, out_features=1024, bias=True)
#         (4): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# attn Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): Attention(
#       (to_qkv): Linear(in_features=1024, out_features=3072, bias=False)
#       (reattn_norm): Sequential(
#         (0): Rearrange('b h i j -> b i j h')
#         (1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
#         (2): Rearrange('b i j h -> b h i j')
#       )
#       (to_out): Sequential(
#         (0): Linear(in_features=1024, out_features=1024, bias=True)
#         (1): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# ff Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=1024, out_features=2048, bias=True)
#         (1): GELU()
#         (2): Dropout(p=0.1, inplace=False)
#         (3): Linear(in_features=2048, out_features=1024, bias=True)
#         (4): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# attn Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): Attention(
#       (to_qkv): Linear(in_features=1024, out_features=3072, bias=False)
#       (reattn_norm): Sequential(
#         (0): Rearrange('b h i j -> b i j h')
#         (1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
#         (2): Rearrange('b i j h -> b h i j')
#       )
#       (to_out): Sequential(
#         (0): Linear(in_features=1024, out_features=1024, bias=True)
#         (1): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# ff Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=1024, out_features=2048, bias=True)
#         (1): GELU()
#         (2): Dropout(p=0.1, inplace=False)
#         (3): Linear(in_features=2048, out_features=1024, bias=True)
#         (4): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# attn Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): Attention(
#       (to_qkv): Linear(in_features=1024, out_features=3072, bias=False)
#       (reattn_norm): Sequential(
#         (0): Rearrange('b h i j -> b i j h')
#         (1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
#         (2): Rearrange('b i j h -> b h i j')
#       )
#       (to_out): Sequential(
#         (0): Linear(in_features=1024, out_features=1024, bias=True)
#         (1): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# ff Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=1024, out_features=2048, bias=True)
#         (1): GELU()
#         (2): Dropout(p=0.1, inplace=False)
#         (3): Linear(in_features=2048, out_features=1024, bias=True)
#         (4): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# attn Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): Attention(
#       (to_qkv): Linear(in_features=1024, out_features=3072, bias=False)
#       (reattn_norm): Sequential(
#         (0): Rearrange('b h i j -> b i j h')
#         (1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
#         (2): Rearrange('b i j h -> b h i j')
#       )
#       (to_out): Sequential(
#         (0): Linear(in_features=1024, out_features=1024, bias=True)
#         (1): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# ff Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=1024, out_features=2048, bias=True)
#         (1): GELU()
#         (2): Dropout(p=0.1, inplace=False)
#         (3): Linear(in_features=2048, out_features=1024, bias=True)
#         (4): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# attn Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): Attention(
#       (to_qkv): Linear(in_features=1024, out_features=3072, bias=False)
#       (reattn_norm): Sequential(
#         (0): Rearrange('b h i j -> b i j h')
#         (1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
#         (2): Rearrange('b i j h -> b h i j')
#       )
#       (to_out): Sequential(
#         (0): Linear(in_features=1024, out_features=1024, bias=True)
#         (1): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# ff Residual(
#   (fn): PreNorm(
#     (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (fn): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=1024, out_features=2048, bias=True)
#         (1): GELU()
#         (2): Dropout(p=0.1, inplace=False)
#         (3): Linear(in_features=2048, out_features=1024, bias=True)
#         (4): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
# )
# x.shape11 transformer shape valuetorch.Size([1, 65, 1024])
# x.shape22 transformer shape valuetorch.Size([1, 1024])
# x.shape33 transformer shape valuetorch.Size([1, 1024])
# torch.Size([1, 1000])
