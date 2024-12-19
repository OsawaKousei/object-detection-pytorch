# python3 -m src.ssd.test.test_dataset

import logging
from logging import Formatter, StreamHandler, getLogger

import pytest
import torch.utils.data as data

import src.mscoco_dataset as msooco_dataset
from src.ssd.utils.util_function import od_collate_fn

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


def test_data_load() -> None:
    dataloader = data.DataLoader(
        msooco_dataset.val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=od_collate_fn,
    )

    for i, (imgs, targets) in enumerate(dataloader):
        logger.debug(f"Batch {i}")
        logger.debug(f"imgs shape: {imgs.shape}")
        logger.debug(f"targets: {targets}")
        assert imgs.shape == (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        assert len(targets) == BATCH_SIZE
        break


# pytestを実行
if __name__ == "__main__":
    pytest.main(["src/ssd/test/test_dataset.py", "-s"])
