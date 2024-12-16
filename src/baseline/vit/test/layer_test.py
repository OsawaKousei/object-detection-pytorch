# python3 -m src.baseline.vit.test.layer_test

import logging
from logging import Formatter, StreamHandler, getLogger

import pytest
import torch

from src.baseline.vit.layers import (
    MultiHeadSelfAttention,
    VitEncorderBlock,
    VitInputLayer,
)

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


def dummy_input() -> torch.Tensor:
    batch_size, channels, height, width = BATCH_SIZE, CHANNELS, 32, 32
    return torch.randn(batch_size, channels, height, width)


def _VitInputLayer(x: torch.Tensor) -> torch.Tensor:
    vit_input_layer = VitInputLayer(
        in_channels=CHANNELS, emb_dim=384, num_patch_row=2, image_size=32
    )
    z_0: torch.Tensor = vit_input_layer(x)

    return z_0


def _MultiHeadSelfAttention(z_0: torch.Tensor) -> torch.Tensor:
    multi_head_self_attention = MultiHeadSelfAttention(
        emb_dim=384, head=3, drop_out=0.1
    )
    out: torch.Tensor = multi_head_self_attention(z_0)

    return out


def _VitEncorderBlock(out: torch.Tensor) -> torch.Tensor:
    vit_encorder_block = VitEncorderBlock(emb_dim=384, head=3, drop_out=0.1)
    z1: torch.Tensor = vit_encorder_block(out)

    return z1


def test_VitInputLayer() -> None:
    x = dummy_input()
    z_0 = _VitInputLayer(x)
    assert z_0.shape == (BATCH_SIZE, 5, 384)


def test_MultiHeadAttention() -> None:
    z_0 = _VitInputLayer(dummy_input())

    out = _MultiHeadSelfAttention(z_0)
    assert out.shape == (BATCH_SIZE, 5, 384)


def test_VitEncorderBlock() -> None:
    z_0 = _VitInputLayer(dummy_input())
    out = _MultiHeadSelfAttention(z_0)

    z1 = _VitEncorderBlock(out)
    assert z1.shape == (BATCH_SIZE, 5, 384)


# pytestを実行
if __name__ == "__main__":
    pytest.main(["src/baseline/vit/test/layer_test.py", "-s"])
