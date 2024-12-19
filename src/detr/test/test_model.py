# python3 -m src.detr.test.test_model

import logging
from logging import Formatter, StreamHandler, getLogger

import pytest
import torch

from src.detr.model import Detr

# ログの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


def dummy_input() -> torch.Tensor:
    return torch.rand(2, 3, 800, 800)


def _Detr(input: torch.Tensor) -> torch.Tensor:
    model = Detr(
        num_classes=91,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    )
    return model(input)


def test_Detr() -> None:
    input = dummy_input()
    output = _Detr(input)
    assert output[0].shape == (2, 100, 92)
    assert output[1].shape == (2, 100, 4)
    logger.info("Detr test passed")


# pytestを実行
if __name__ == "__main__":
    pytest.main(["src/detr/test/test_model.py", "-s"])
