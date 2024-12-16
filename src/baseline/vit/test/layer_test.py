# python3 -m src.baseline.vit.test.layer_test

import logging
from logging import Formatter, StreamHandler, getLogger

import pytest
import torch

from src.baseline.vit.layers import VitInputLayer

# ログの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


def test_VitInputLayer() -> None:
    batch_size, channels, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    vit_input_layer = VitInputLayer(
        in_channels=channels, emb_dim=384, num_patch_row=2, image_size=32
    )
    z_0 = vit_input_layer(x)
    assert z_0.shape == (batch_size, 5, 384)


# pytestを実行
if __name__ == "__main__":
    pytest.main(["src/baseline/vit/test/layer_test.py", "-s"])
