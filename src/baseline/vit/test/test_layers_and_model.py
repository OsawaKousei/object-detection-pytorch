# python3 -m src.baseline.vit.test.test_layers_and_model

import logging
from logging import Formatter, StreamHandler, getLogger

import pytest
import torch

from src.baseline.vit.layers import (
    MultiHeadSelfAttention,
    VitEncorderBlock,
    VitInputLayer,
)
from src.baseline.vit.vit_model import Vit

# ログの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

CHANNELS = 3
BATCH_SIZE = 2

IMG_SIZE = 32
NUM_CLASSES = 10
NUM_PATCH_ROW = 2
NUM_BLOCKS = 7
EMB_DIM = 384
HIDDEN_DIM = EMB_DIM * 4
DROP_OUT = 0.0
HEAD = 8


def dummy_input() -> torch.Tensor:
    batch_size, channels, height, width = BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE
    return torch.randn(batch_size, channels, height, width)


def _VitInputLayer(x: torch.Tensor) -> torch.Tensor:
    vit_input_layer = VitInputLayer(
        in_channels=CHANNELS,
        emb_dim=EMB_DIM,
        num_patch_row=NUM_PATCH_ROW,
        image_size=IMG_SIZE,
    )
    z_0: torch.Tensor = vit_input_layer(x)

    return z_0


def _MultiHeadSelfAttention(z_0: torch.Tensor) -> torch.Tensor:
    multi_head_self_attention = MultiHeadSelfAttention(
        emb_dim=EMB_DIM, head=HEAD, drop_out=DROP_OUT
    )
    out: torch.Tensor = multi_head_self_attention(z_0)

    return out


def _VitEncorderBlock(out: torch.Tensor) -> torch.Tensor:
    vit_encorder_block = VitEncorderBlock(emb_dim=EMB_DIM, head=HEAD, drop_out=DROP_OUT)
    z1: torch.Tensor = vit_encorder_block(out)

    return z1


def _Vit(x: torch.Tensor) -> torch.Tensor:
    vit = Vit(
        in_channels=CHANNELS,
        num_classes=NUM_CLASSES,
        emb_dim=EMB_DIM,
        num_patch_row=NUM_PATCH_ROW,
        image_size=IMG_SIZE,
        num_blocks=NUM_BLOCKS,
        head=HEAD,
        hidden_dim=HIDDEN_DIM,
        drop_out=DROP_OUT,
    )
    pred: torch.Tensor = vit(x)

    return pred


def test_VitInputLayer() -> None:
    x = dummy_input()
    z_0 = _VitInputLayer(x)
    assert z_0.shape == (BATCH_SIZE, NUM_PATCH_ROW**2 + 1, EMB_DIM)


def test_MultiHeadAttention() -> None:
    z_0 = _VitInputLayer(dummy_input())

    out = _MultiHeadSelfAttention(z_0)
    assert out.shape == (BATCH_SIZE, NUM_PATCH_ROW**2 + 1, EMB_DIM)


def test_VitEncorderBlock() -> None:
    z_0 = _VitInputLayer(dummy_input())
    out = _MultiHeadSelfAttention(z_0)

    z1 = _VitEncorderBlock(out)
    assert z1.shape == (BATCH_SIZE, NUM_PATCH_ROW**2 + 1, EMB_DIM)


def test_Vit() -> None:
    x = dummy_input()
    pred = _Vit(x)
    assert pred.shape == (BATCH_SIZE, NUM_CLASSES)


# pytestを実行
if __name__ == "__main__":
    pytest.main(["src/baseline/vit/test/test_layers_and_model.py", "-s"])
