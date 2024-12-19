# python3 -m src.detr.train

import logging
import os
import random
from logging import Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.init as init
import torch.utils.data as data
from torch import nn, optim
from tqdm import tqdm

import src.mscoco_dataset as msooco_dataset
from src.mscoco_dataset import od_collate_fn
from src.ssd.model import SSD
from src.ssd.utils.ssd_model import MultiBoxLoss

# ログの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


class SSDTrainer:
    def __init__(
        self,
        ws_dir: str,
        train_dataset: data.Dataset,
        valid_dataset: data.Dataset,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        batch_size: int = 32,
        num_epochs: int = 50,
    ) -> None:
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=od_collate_fn,
        )

        self.valid_dataloader = data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=od_collate_fn,
        )

        self.ws_dir = ws_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        logger.info("Trainer initialized")

    def train(self) -> None:
        self.net.to(self.device)
        # 学習と検証
        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            train_loss: list[float] = []
            train_acc: list[float] = []
            val_loss: list[float] = []
            val_acc: list[float] = []
            logs: list[dict] = []

            for epoch in tglobal:
                # 学習
                with tqdm(self.train_dataloader, desc="Train", leave=False) as t:
                    sum_loss = 0.0  # lossの合計
                    sum_correct = 0  # 正解率の合計
                    sum_total = 0  # dataの数の合計

                    for inputs, labels in t:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer.zero_grad()
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels)
                        sum_loss += loss.item()
                        sum_total += labels.size(0)
                        predicted = torch.argmax(outputs, dim=1)
                        gt = torch.argmax(labels, dim=1)
                        sum_correct += (predicted == gt).sum().item()
                        loss.backward()
                        self.optimizer.step()

                        t.set_postfix(loss=loss.item())

                    loss = (
                        sum_loss * self.batch_size / len(self.train_dataloader.dataset)
                    )
                    acc = float(sum_correct / sum_total)
                    train_loss.append(loss)
                    train_acc.append(acc)

                # 検証
                with tqdm(self.valid_dataloader, desc="Valid", leave=False) as v:
                    sum_loss = 0.0
                    sum_correct = 0
                    sum_total = 0

                    with torch.no_grad():
                        for inputs, labels in v:
                            inputs, labels = inputs.to(self.device), labels.to(
                                self.device
                            )
                            self.optimizer.zero_grad()
                            outputs = self.net(inputs)
                            loss = self.criterion(outputs, labels)
                            sum_loss += loss.item()
                            sum_total += labels.size(0)
                            predicted = torch.argmax(outputs, dim=1)
                            gt = torch.argmax(labels, dim=1)
                            sum_correct += (predicted == gt).sum().item()

                            v.set_postfix(loss=loss.item())

                        loss = (
                            sum_loss
                            * self.batch_size
                            / len(self.valid_dataloader.dataset)
                        )
                        acc = float(sum_correct / sum_total)
                        val_loss.append(loss)
                        val_acc.append(acc)

                tglobal.set_postfix(loss=loss, acc=acc)

                logs.append(
                    {
                        "epoch": epoch,
                        "train_loss": np.mean(loss),
                        "val_loss": np.mean(val_loss),
                    }
                )

                self.save_logs(logs)

                # モデルを保存
                if (epoch + 1) % 5 == 0:
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(
                            self.ws_dir, "model", "model_{}.pth".format(epoch)
                        ),
                    )

    def save_logs(self, logs: list) -> None:
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(self.ws_dir, "train_logs.csv"))

        plt.plot(df["train_loss"], label="train_loss")
        plt.plot(df["val_loss"], label="val_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Logs")
        plt.savefig(os.path.join(self.ws_dir, "train_logs.png"))
        plt.close()


if __name__ == "__main__":

    # 乱数のシードを設定
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # SSD300の設定
    ssd_cfg = {
        "num_classes": 21,  # 背景クラスを含めた合計クラス数
        "input_size": 300,  # 画像の入力サイズ
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        "feature_maps": [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        "steps": [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        "min_sizes": [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        "max_sizes": [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    # SSDネットワークモデル
    net = SSD(phase="train", cfg=ssd_cfg)

    # SSDの初期の重みを設定
    # ssdのvgg部分に重みをロードする
    vgg_weights = torch.load("./weights/vgg16_reducedfc.pth")
    net.vgg.load_state_dict(vgg_weights)

    # ssdのその他のネットワークの重みはHeの初期値で初期化

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:  # バイアス項がある場合
                nn.init.constant_(m.bias, 0.0)

    # Heの初期値を適用
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    print("ネットワーク設定完了：学習済みの重みをロードしました")

    # 損失関数の設定
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

    # 最適化手法の設定
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    trainer = SSDTrainer(
        ws_dir=".",
        train_dataset=msooco_dataset.train_dataset,
        valid_dataset=msooco_dataset.val_dataset,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        batch_size=32,
        num_epochs=50,
    )

    trainer.train()
