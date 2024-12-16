import os

import torch
import torch.utils.data as data
from torch import nn, optim
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.baseline.ILSVRC_dataset import valid_dataset as dataset_
from src.baseline.vit.vit_model import Vit


class VitTrainer:
    def __init__(
        self,
        ws_dir: str,
        dataset: ImageFolder,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        batch_size: int = 64,
        num_epochs: int = 10,
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

    def train(self) -> None:
        # 学習と検証
        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            train_loss: list[float] = []
            train_acc: list[float] = []
            val_loss: list[float] = []
            val_acc: list[float] = []

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
                        _, predicted = outputs.max(1)
                        sum_total += labels.size(0)
                        sum_correct += (predicted == labels).sum().item()
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
                            _, predicted = outputs.max(1)
                            sum_total += labels.size(0)
                            sum_correct += (predicted == labels).sum().item()

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

                # モデルを保存
                if (epoch + 1) % 5 == 0:
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(
                            self.ws_dir, "model", "model_{}.pth".format(epoch)
                        ),
                    )


if __name__ == "__main__":
    net = Vit(
        in_channels=3,
        num_classes=1000,
        emb_dim=768,
        num_patch_row=16,
        image_size=224,
        num_blocks=12,
        head=12,
        hidden_dim=3072,
        drop_out=0.1,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ws_dir = "src/baseline/vit/result"
    dataset = dataset_
    trainer = VitTrainer(
        ws_dir=ws_dir,
        dataset=dataset,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
    trainer.train()
