# python3 -m src.ssd.test.test_model

import logging
from logging import Formatter, StreamHandler, getLogger

import pytest
import torch

from src.ssd.loss_function import MultiBoxLoss
from src.ssd.model import SSD

# ログの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


def dummy_input() -> torch.Tensor:
    return torch.rand(2, 3, 300, 300)


def dummy_target() -> torch.Tensor:
    return torch.rand(2, 2, 5)


def _SSD(input: torch.Tensor) -> torch.Tensor:
    # SSD300の設定
    ssd_cfg = {
        "num_classes": 92,  # 背景クラスを含めた合計クラス数
        "input_size": 300,  # 画像の入力サイズ
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        "feature_maps": [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        "steps": [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        "min_sizes": [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        "max_sizes": [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    # SSDネットワークモデル
    model = SSD(phase="train", cfg=ssd_cfg)

    return model(input)


def test_SSD() -> None:
    input = dummy_input()
    output = _SSD(input)
    logger.debug(f"loc shape: {output[0].shape}")
    logger.debug(f"conf shape: {output[1].shape}")
    logger.debug(f"dbox_list shape: {output[2].shape}")
    assert output[0].shape == (2, 8732, 4)
    assert output[1].shape == (2, 8732, 92)
    assert output[2].shape == (8732, 4)


def test_loss() -> None:
    input = dummy_input()
    output = _SSD(input)
    loss_function = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device="cpu")
    loss = loss_function(output, dummy_target())
    logger.debug(f"loc loss: {loss[0]}")
    logger.debug(f"conf loss: {loss[1]}")
    assert loss[0].shape == ()
    assert loss[1].shape == ()


# pytestを実行
if __name__ == "__main__":
    pytest.main(["src/ssd/test/test_model.py", "-s"])
