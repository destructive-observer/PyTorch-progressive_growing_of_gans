from models.base_model import *


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class Norm_Linear(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        # self.net = nn.Linear(in_dim,out_dim)
        self.norm =  nn.Linear(in_dim,out_dim)
        he_init(self.norm,'leakyrelu',0.2)
        self.scalelayer = Trans_WScaleLayer(self.norm)

    def forward(self,x):
        # print('x shape{}'.format(x.shape))
        # x1 = self.net(x)
        # print(self.net)
        # print('x1 net shape{}'.format(x1.shape))
        # print(self.norm)
        x1 = self.norm(x)
        # print('x2 norm shape{}'.format(x2.shape))
        x3 = self.scalelayer(x1)
        # print('x3 scale shape{}'.format(x2.shape))
        return x3

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
class SpectralNorm(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.norm = nn.utils.spectral_norm(fn)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            # nn.utils.spectral_norm(nn.Linear(dim, hidden_dim)),
            Norm_Linear(dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Dropout(dropout),
            # nn.utils.spectral_norm(nn.Linear(hidden_dim, dim)),
            Norm_Linear(hidden_dim, dim),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)

        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 256, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3)

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
        print('self.to_qkv {}'.format(self.to_qkv))
        print('self.to_qkv x shape {},{},{},{}'.format(b, n, _, h))
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        # attn=dots
        # re-attention

        # attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        # attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

def G_transformer(incoming, dim, heads,dim_head, dropout,mlp_dim,curr_patchsize,channel,curr_dim,curr_num,num_channels,to_dim,
        to_sequential=True, use_wscale=True, use_pixelnorm=True):   
    layers = incoming
    layers += [#attention input b (h w) (p1 p2 c)
        Rearrange('b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)', p1 = curr_patchsize, p2 = curr_patchsize,c=channel,h=curr_num,w=curr_num),
        # Residual(
            Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
            # ),
            #    Residual(                
                #  FeedForward(dim, mlp_dim, dropout = dropout),
                 FeedForward(dim, mlp_dim, dropout = dropout),
                #  ),
                # nn.LeakyReLU(negative_slope=0.2),
                 Rearrange('b (h w) (p1 p2 c) -> b c (p1 h) (p2 w)', p1 = curr_patchsize, p2 = curr_patchsize,c=channel,h=curr_num,w=curr_num)     
                #    nn.GELU(),
                #    )
               ]
            #    Norm_Linear(dim, to_dim),
            #    Norm_Linear(2048, 2048),
            #    Norm_Linear(2048, to_dim),
            #    Rearrange('b (h w) (p1 p2 c) -> b c (p1 h) (p2 w)', p1 = curr_patchsize, p2 = curr_patchsize,c=num_channels,h=curr_num,w=curr_num)]
    # layers1=[]
    # output -->b c h w
    # he_init(layers[-1], init, param)  # init layers
    # if use_wscale:
    #     for i,value in enumerate(layers):
    #         print('G_Trans i {}'.format(value))
    #         layers1 += [Trans_WScaleLayer(value)]    
    # layers += [nonlinearity]
    # if use_batchnorm:
        # layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    # layers1 = layers
    # layers = incoming+layers1
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

def G_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
        to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
    layers = incoming

    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_batchnorm:
        layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

def NIN_transformer(incoming, dim, heads,dim_head, dropout,mlp_dim,curr_patchsize,out_channel,curr_num,curr_dim,to_dim,in_channels,
        to_sequential=True, use_wscale=True):
    layers = incoming
    layers += [
        Rearrange('b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)', p1 = curr_patchsize, p2 = curr_patchsize,c=in_channels,h=curr_num,w=curr_num),
        # Residual(
        Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
        # ),
            #    Residual(
                FeedForward(dim, mlp_dim, dropout = dropout),
                # ),
                #    ),
               Norm_Linear(dim, to_dim),
               nn.LeakyReLU(negative_slope=0.2)
               ,
    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = curr_patchsize, p2 = curr_patchsize,c=out_channel,h=curr_num,w=curr_num)

            #    Norm_Linear(2048, 2048),
            #    Norm_Linear(2048, to_dim),
            # nn.Linear(curr_dim, curr_dim),
               ]
    # layers1=[]
    # if use_wscale:
    #     for i,value in enumerate(layers):
    #         layers1 += [Trans_WScaleLayer(value)]
    # layers1 = layers
    # layers = incoming+layers1
    # if not (nonlinearity == 'linear'):
        # layers += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

def NINLayer(incoming, in_channels, out_channels, nonlinearity, init, param=None, 
            to_sequential=True, use_wscale=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    if not (nonlinearity == 'linear'):
        layers += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Generator(nn.Module):
    def __init__(self, 
                num_channels        = 3,        # Overridden based on dataset.
                resolution          = 32,       # Overridden based on dataset.
                first_resolution    = 4,# Overridden based on dataset.
                label_size          = 0,        # Overridden based on dataset.
                fmap_base           = 4096,
                fmap_decay          = 1.0,
                fmap_max            = 256,
                dim = 256,#可变参数
                # dim = 2048,#可变参数
                base_size = 2,
                max_patch = 32,
                channel = 64,
                # channel = 512
                heads = 16,
                dim_head = 64,
                # mlp_dim = 4096,#可变
                mlp_dim = 256,
                dropout = 0.0,
                latent_size         = None,
                normalize_latents   = True,
                use_wscale          = True,
                use_pixelnorm       = True,
                use_leakyrelu       = True,
                use_batchnorm       = False,
                tanh_at_end         = None):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.latent_size = latent_size
        self.normalize_latents = normalize_latents
        self.use_wscale = use_wscale
        self.use_pixelnorm = use_pixelnorm
        self.use_leakyrelu = use_leakyrelu
        self.use_batchnorm = use_batchnorm
        self.tanh_at_end = tanh_at_end
        self.curr_resol =first_resolution

        R = int(np.log2(resolution))
        assert resolution == 2**R and resolution >= 4
        if latent_size is None: 
            latent_size = self.get_nf(0)

        # negative_slope = 0.2
        # act = nn.LeakyReLU(negative_slope=negative_slope) if self.use_leakyrelu else nn.ReLU()
        # iact = 'leaky_relu' if self.use_leakyrelu else 'relu'
        # output_act = nn.Tanh() if self.tanh_at_end else 'linear'
        # output_iact = 'tanh' if self.tanh_at_end else 'linear'

        pre = None
        lods = nn.ModuleList()
        nins = nn.ModuleList()
        layers = []

        if self.normalize_latents:
            pre = PixelNormLayer()

        if self.label_size:
            layers += [ConcatLayer()]

        # layers += [ReshapeLayer([int(latent_size/(first_resolution**2)), first_resolution, first_resolution])]
        # layers = G_conv(layers, latent_size, self.get_nf(1), 4, 3, act, iact, negative_slope, 
                    # False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm) 
                    #第一层换成MLP 从512--8192
        # layers += [nn.Linear(latent_size, 1024)]
        # layers += [Norm_Linear(latent_size, 8192),nn.LeakyReLU(negative_slope=0.2)]#b*8192 --> b*512*4*4
        layers += [Norm_Linear(latent_size, 1024)]
        # ,nn.LeakyReLU(negative_slope=0.2)]#b*8192 --> b*512*4*4

        # layers += [Norm_Linear(2048, 4096),nn.LeakyReLU(negative_slope=0.2)]
        # layers += [Norm_Linear(4096, 8192),nn.LeakyReLU(negative_slope=0.2)]
        # layers += [Norm_Linear(4096, 8192)]
        # net = G_conv(layers, latent_size, self.get_nf(1), 3, 1, act, iact, negative_slope, 
                    # True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)  # first block
        # layers += [nn.Linear(4096, 4096)]
        ## input reshape b * dim --> b c h w

        curr_patchsize,curr_dim,curr_num = self.get_patch_dim(1,base_size,max_patch,channel,first_resolution)
        print('curr_dim {}, curr_patchsize {}'.format(curr_dim,curr_patchsize))
        ## b * 4096 -->b*512*4*4-->b*(2*2)*(512*2*2)
        layers +=[Rearrange('b (c p1 h p2 w) -> b c (h p1) (w p2)', p1 = curr_patchsize, p2 = curr_patchsize,h=curr_num,w=curr_num)]
        # layers += [#transformer input b c h w
        # nn.Sequential(
        #     # Rearrange('b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)', p1 = curr_patchsize, p2 = curr_patchsize,c=latent_size//2,h=curr_num,w=curr_num),
        #     Norm_Linear(curr_dim, dim),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     # Norm_Linear(256, 512),
        #     # Norm_Linear(512, dim),
        # )]       
        net = G_transformer(layers,dim,heads,dim_head,dropout,mlp_dim,curr_patchsize,channel,curr_dim,curr_num,num_channels,num_channels*curr_patchsize*curr_patchsize)
        
        # lods.append(to_patch_embedding)
        # net = 
        lods.append(net)
        # nins.append(NINLayer([], self.get_nf(1), self.num_channels, output_act, output_iact, None, True, self.use_wscale))  # to_rgb layer
        nins.append(NIN_transformer([], dim, heads, dim_head, dropout, mlp_dim,curr_patchsize,num_channels,curr_num,curr_dim,num_channels*curr_patchsize*curr_patchsize,channel))
 
        # nins.append(to_patch_image)
        for I in range(2, R):  # following blocks
            # ic, oc = self.get_nf(I-1), self.get_nf(I)
            curr_patchsize,curr_dim,curr_num = self.get_patch_dim(I,base_size,max_patch,num_channels,first_resolution)
            last_patchsize,last_dim,last_num = self.get_patch_dim(I-1,base_size,max_patch,num_channels,first_resolution)
            print('following curr_dim {}, curr_patchsize {}'.format(curr_dim,curr_patchsize))
            layers = [nn.Upsample(scale_factor=2, mode='bicubic')]  # upsample
            # layers = [nn.PixelShuffle(2)]  # 2D->upsample mul 2
            # layers +=[Rearrange('b n (p1 p2 c d) -> b (n d) (p1 p2 c)',p1=last_patchsize,p2=last_patchsize,d=4)]
            layers = G_transformer(layers,dim*(4**(I-1)),heads,dim_head,dropout,mlp_dim*(4**(I-1)),curr_patchsize,channel,curr_dim,curr_num,num_channels,curr_dim)
            # layers = G_conv(layers, ic, oc, 3, 1, act, iact, negative_slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            
            # net = G_conv(layers, oc, oc, 3, 1, act, iact, negative_slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            net = layers
            lods.append(net)
            nins.append(NIN_transformer([], dim*(4**(I-1)), heads, dim_head, dropout, mlp_dim*(4**(I-1)),curr_patchsize,num_channels,curr_num,curr_dim,curr_dim,channel))
            # nins.append(NINLayer([], oc, self.num_channels, output_act, output_iact, None, True, self.use_wscale))  # to_rgb layer

        self.output_layer = GSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)
    def get_patch_dim(self,stage,base_size,max_patch,channel,first_resol):
        patch_size = min(max_patch,base_size**(stage))#stage 1 --> xx base_size=2
        dim = (patch_size**2) * channel
        curr_num = first_resol*(2**(stage-1))//patch_size
        return int(patch_size),int(dim),int(curr_num)


    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        return self.output_layer(x, y, cur_level, insert_y_at)


def D_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
        to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
    layers = incoming
    if use_gdrop:
        layers += [GDropLayer(**gdrop_param)]
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    # print(layers)
    if use_layernorm:
        # layers += [nn.BatchNorm2d(out_channels)]
        layers += [LayerNormLayer(layers[-1],'leaky_relu')]  # TODO: requires incoming layer

    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Discriminator(nn.Module):
    def __init__(self, 
                num_channels    = 1,        # Overridden based on dataset.
                resolution      = 32,       # Overridden based on dataset.
                label_size      = 0,        # Overridden based on dataset.
                fmap_base       = 4096,
                fmap_decay      = 1.0,
                fmap_max        = 256,
                mbstat_avg      = 'all',
                mbdisc_kernels  = None,
                use_wscale      = True,
                use_gdrop       = True,
                use_layernorm   = False,
                sigmoid_at_end  = False):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.mbstat_avg = mbstat_avg
        self.mbdisc_kernels = mbdisc_kernels
        self.use_wscale = use_wscale
        self.use_gdrop = use_gdrop
        self.use_layernorm = use_layernorm
        self.sigmoid_at_end = sigmoid_at_end

        R = int(np.log2(resolution))
        assert resolution == 2**R and resolution >= 4
        gdrop_strength = 0.0

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope)
        # input activation
        iact = 'leaky_relu'
        # output activation
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_iact = 'sigmoid' if self.sigmoid_at_end else 'linear'
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

        nins = nn.ModuleList()
        lods = nn.ModuleList()
        pre = None

        nins.append(NINLayer([], self.num_channels, self.get_nf(R-1), act, iact, negative_slope, True, self.use_wscale))

        for I in range(R-1, 1, -1):
            ic, oc = self.get_nf(I), self.get_nf(I-1)
            net = D_conv([], ic, ic, 3, 1, act, iact, negative_slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, 
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            lods.append(nn.Sequential(*net))
            # nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            nin = []
            nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
            nins.append(nin)

        net = []
        ic = oc = self.get_nf(1)
        if self.mbstat_avg is not None:
            net += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)]
            ic += 1
        net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, 
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        net = D_conv(net, oc, self.get_nf(0), 4, 0, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)

        # Increasing Variation Using MINIBATCH Standard Deviation
        if self.mbdisc_kernels:
            net += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]

        oc = 1 + self.label_size
        # lods.append(NINLayer(net, self.get_nf(0), oc, 'linear', 'linear', None, True, self.use_wscale))
        lods.append(NINLayer(net, self.get_nf(0), oc, output_act, output_iact, None, True, self.use_wscale))

        self.output_layer = DSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(x, y, cur_level, insert_y_at)


# class AutoencodingDiscriminator(nn.Module):
#     def __init__(self, 
#                 num_channels    = 1,        # Overridden based on dataset.
#                 resolution      = 32,       # Overridden based on dataset.
#                 fmap_base       = 4096,
#                 fmap_decay      = 1.0,
#                 fmap_max        = 256,
#                 tanh_at_end     = False):
#         super(AutoencodingDiscriminator, self).__init__()
#         self.num_channels = num_channels
#         self.resolution = resolution
#         self.fmap_base = fmap_base
#         self.fmap_decay = fmap_decay
#         self.fmap_max = fmap_max
#         self.tanh_at_end = tanh_at_end

#         R = int(np.log2(resolution))
#         assert resolution == 2**R and resolution >= 4
        
#         negative_slope = 0.2
#         act = nn.LeakyReLU(negative_slope=negative_slope)
#         iact = 'leaky_relu'
#         output_act = nn.Tanh() if self.tanh_at_end else 'linear'
#         output_iact = 'tanh' if self.tanh_at_end else 'linear'

#         nins = nn.ModuleList()
#         lods = nn.ModuleList()
#         pre = None

#         for I in range(R, 1, -1):
#             ic, oc = self.get_nf(I), self.get_nf(I-1)
#             nins.append(NINLayer([], self.num_channels, ic, act, iact, negative_slope, True, True))  # from_rgb layer

#             net = [nn.Conv2d(ic, oc, 3, 1, 1), act]
#             net += [nn.BatchNorm2d(oc), nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
#             he_init(net[0], iact, negative_slope)
#             lods.append(nn.Sequential(*net))

#         for I in range(2, R+1):
#             ic, oc = self.get_nf(I-1), self.get_nf(I)
#             net = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ic, oc, 3, 1, 1), act, nn.BatchNorm2d(oc)]
#             he_init(net[1], iact, negative_slope)
#             lods.append(nn.Sequential(*net))
#             nins.append(NINLayer([], oc, self.num_channels, output_act, output_iact, None, True, True))  # to_rgb layer

#         self.output_layer = AEDSelectLayer(pre, lods, nins)

#     def get_nf(self, stage):
#         return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

#     def forward(self, x, cur_level=None):
#         return self.output_layer(x, cur_level)
