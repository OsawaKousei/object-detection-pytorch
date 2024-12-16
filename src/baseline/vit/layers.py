import torch
import torch.nn as nn


class VitInputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,  # assume square patches
        image_size: int = 32,  # assume square image
    ):
        super(VitInputLayer, self).__init__()
        self.image_size = image_size
        self.num_patch_row = num_patch_row
        self.emb_dim = emb_dim
        self.in_channels = in_channels

        # num of patches
        self.num_patches = num_patch_row**2

        # patch size
        self.patch_size = int(image_size // num_patch_row)

        # patch embedding
        self.patch_emb_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch embedding
        # (B, C, H, W) -> (B, D, H/P, W/P)
        z_0: torch.Tensor = self.patch_emb_layer(x)

        # flatten patches
        # (B, D, H/P, W/P) -> (B, N_patches, D)
        z_0 = z_0.flatten(2).transpose(1, 2)

        # add cls token
        # (B, N_patches, D) -> (B, N, D) N = N_patches + 1
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(z_0.shape[0], 1, 1)), z_0], dim=1
        )

        # add positional embedding
        # (B, N, D) -> (B, N, D)
        z_0 = z_0 + self.pos_emb

        return z_0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int = 384, head: int = 3, drop_out: float = 0):
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head = head
        self.drop_out = drop_out

        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5

        # query, key, value
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # drop out
        self.attention_drop = nn.Dropout(drop_out)

        # output
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(drop_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, _ = z.size()

        # query, key, value
        q: torch.Tensor = self.w_q(z)
        k: torch.Tensor = self.w_k(z)
        v: torch.Tensor = self.w_v(z)

        # split
        # (B, N, D) -> (B, N, H, D/H)
        q = q.view(batch_size, num_patches, self.head, self.head_dim)
        k = k.view(batch_size, num_patches, self.head, self.head_dim)
        v = v.view(batch_size, num_patches, self.head, self.head_dim)

        # attention
        # (B, N, H, D/H) -> (B, H, N, D/H)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention score
        # (B, H, N, D/H) -> (B, H, N, D/H)
        K_T = k.transpose(2, 3)

        # (B, H, N, D/H) @ (B, H, D/H, N) -> (B, H, N, N)
        dots = (q @ K_T) / self.sqrt_dh

        # softmax
        attn = torch.softmax(dots, dim=-1)
        # drop out
        attn = self.attention_drop(attn)

        # weighted sum
        # (B, H, N, N) @ (B, H, N, D/H) -> (B, H, N, D/H)
        out: torch.Tensor = attn @ v
        # (B, H, N, D/H) -> (B, N, H, D/H)
        out = out.transpose(1, 2)
        # (B, N, H, D/H) -> (B, N, D)
        out = out.reshape(batch_size, num_patches, self.emb_dim)

        # output
        out = self.w_o(out)

        return out
