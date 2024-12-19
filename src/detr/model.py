from logging import getLogger

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = getLogger("__main__").getChild(__name__)


class Detr(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        nheads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
    ):
        super().__init__()
        self.backborn = nn.Sequential(
            *list(models.resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-2]
        )
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        logger.debug(f"inputs shape: {inputs.shape}")  # torch.Size([2, 3, 300, 300])

        x: torch.Tensor = self.backborn(inputs)
        logger.debug(
            f"backborn output shape: {x.shape}"
        )  # torch.Size([2, 2048, 10, 10])

        h: torch.Tensor = self.conv(x)
        logger.debug(f"conv output shape: {h.shape}")  # torch.Size([2, 256, 10, 10])

        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        logger.debug(f"pos shape: {pos.shape}")  # torch.Size([100, 1, 256])

        src: torch.Tensor = pos + h.flatten(2).permute(2, 0, 1)
        tgt: torch.Tensor = self.query_pos.unsqueeze(1).repeat(1, inputs.shape[0], 1)
        logger.debug(f"src shape: {src.shape}")  # torch.Size([100, 2, 256])
        logger.debug(f"tgt shape: {tgt.shape}")  # torch.Size([100, 2, 256])

        out: torch.Tensor = self.transformer(
            src=src,
            tgt=tgt,
        )
        logger.debug(
            f"transformer output shape: {out.shape}"
        )  # torch.Size([2, 256, 10 ,10])

        pred_class = self.linear_class(out)  # torch.Size([2, 100, 91])
        pred_box = self.linear_bbox(out).sigmoid()  # torch.Size([2, 100, 4])
        logger.debug(f"pred_class shape: {pred_class.shape}")
        logger.debug(f"pred_box shape: {pred_box.shape}")

        return {"pred_logits": pred_class, "pred_boxes": pred_box}
