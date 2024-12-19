from typing import Any

import scipy.io
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

valid_root = "/home/kousei/dataset/image_datasets/ILSVRC/2012/ILSVRC2012_img_val_for_ImageFolder"  # 検証データのフォルダ
metadata_path = "/home/kousei/dataset/image_datasets/ILSVRC/2012/ILSVRC2012_devkit_t12/data/meta.mat"

valid_transform = transforms.Compose(
    [
        transforms.Resize(224),  # 1辺が224ピクセルの正方形に変換
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Tensor行列に変換
    ]
)
valid_dataset = ImageFolder(
    root=valid_root, transform=valid_transform  # 画像が保存されているフォルダのパス
)  # Tensorへの変換

# ラベルIDの変換用dict
meta = scipy.io.loadmat(metadata_path, squeeze_me=True)
ilsvrc2012_id_to_wnid = {
    meta["synsets"][1][1]: meta["synsets"][i][0] for i in range(0, 1000)
}


def get_one_hot_vector(label: Any) -> torch.Tensor:
    vector = torch.zeros(1000)
    idx = ilsvrc2012_id_to_wnid[label]
    vector[idx - 1] = 1.0

    return vector


class ILSVRC2012Dataset(data.Dataset):
    def __init__(
        self,
        root_dir: str = valid_root,
        transform: Any = valid_transform,
        wnid_to_index: Any = ilsvrc2012_id_to_wnid,
    ) -> None:
        self.image_folder = ImageFolder(root=root_dir, transform=transform)
        self.wnid_to_index = wnid_to_index

        if self.wnid_to_index is None:
            raise ValueError("wnid_to_index マッピングが必要です。")

    def __len__(self) -> int:
        return len(self.image_folder)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label_idx = self.image_folder[idx]
        one_hot_label = torch.zeros(1000)
        one_hot_label[label_idx] = 1.0

        return img, one_hot_label


if __name__ == "__main__":
    valid_dataset = ILSVRC2012Dataset(
        root_dir=valid_root,
        transform=valid_transform,
        wnid_to_index=ilsvrc2012_id_to_wnid,
    )
    print(valid_dataset[0])
    # 動作確認
    data_loader = data.DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
    )
    for imgs, labels in data_loader:
        print(imgs.size(), labels.size())
        break
