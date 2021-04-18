import torch
from torch import nn, einsum
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ConvEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7, stride=2, padding=3, pool_kernel_size=3, pool_stride=2,
                 pool_padding=1):
        super(ConvEmbed, self).__init__()



        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                    padding=padding, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            Rearrange('b d h w -> b (h w) d')
            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.conv_layers(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class CompactTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim=768, depth=12, heads=12, pool='cls', in_channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1, scale_dim=4, conv_embed=False):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        if conv_embed:
            self.to_patch_embedding = ConvEmbed(in_channels, dim)
            num_patches = self.to_patch_embedding.sequence_length()
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)



        self.pool = nn.Linear(dim, 1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.apply(self.init_weight)



    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        g = self.pool(x)
        xl = F.softmax(g, dim=1)
        x = einsum('b n l, b n d -> b l d', xl, x)

        return self.mlp_head(x.squeeze(-2))

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    cvt = CompactTransformer(224, 16, 1000)

    parameters = filter(lambda p: p.requires_grad, cvt.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters in CVT: %.3fM' % parameters)

    out = cvt(img)

    print("Shape of out :", out.shape)  # [B, num_classes]

    cct = CompactTransformer(224, 16, 1000, conv_embed=True)

    parameters = filter(lambda p: p.requires_grad, cct.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters in CCT: %.3fM' % parameters)

    out = cct(img)

    print("Shape of out :", out.shape)  # [B, num_classes]