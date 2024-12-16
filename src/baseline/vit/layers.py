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
