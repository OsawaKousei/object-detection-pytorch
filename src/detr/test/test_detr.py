# python3 -m src.detr.test.test_model

import logging
from logging import Formatter, StreamHandler, getLogger

import pytest
import torch

from src.detr.loss import SetCriterion
from src.detr.matcher import HungarianMatcher
from src.detr.model import Detr

# ログの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

BATCH_SIZE = 2
IMG_SIZE = 300
COLOR_CHANNELS = 3
NUM_CLASSES = 91


def dummy_input() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, COLOR_CHANNELS, IMG_SIZE, IMG_SIZE)


def dummy_target() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor(
                [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32
            ),
        }
        for _ in range(BATCH_SIZE)
    ]


def dummy_prediction() -> dict[str, torch.Tensor]:
    return {
        "pred_logits": torch.rand(BATCH_SIZE, 100, NUM_CLASSES + 1),
        "pred_boxes": torch.rand(BATCH_SIZE, 100, 4),
    }


def _Detr(input: torch.Tensor) -> torch.Tensor:
    model = Detr(
        num_classes=NUM_CLASSES,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    )
    return model(input)


def test_Detr() -> None:
    input = dummy_input()
    output = _Detr(input)
    assert output["pred_logits"].shape == (100, 2, 92)
    assert output["pred_boxes"].shape == (100, 2, 4)
    logger.info("Detr test passed")


def test_loss() -> None:
    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        eos_coef=0.1,
        losses=["labels", "boxes"],
    )
    output, target = dummy_prediction(), dummy_target()
    loss = criterion(output, target)
    assert loss is not None
    logger.info("Loss test passed")


# pytestを実行
if __name__ == "__main__":
    pytest.main(["src/detr/test/test_detr.py", "-s"])
