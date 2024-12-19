import logging
import os
from logging import Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch import nn, optim
from tqdm import tqdm

# ログの設定
logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setFormatter(handler_format)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


class VitTrainer:
    def __init__(
        self,
        ws_dir: str,
        dataset: data.Dataset,
        num_classes: int,
        num_queries: int,
        aux_loss: bool,
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

        # datasetをランダムに分割
        valid_size = int(0.1 * len(dataset))
        train_size = len(dataset) - valid_size
        train_dataset, valid_dataset = data.random_split(
            dataset, [train_size, valid_size]
        )

        self.train_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        self.valid_dataloader = data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
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
