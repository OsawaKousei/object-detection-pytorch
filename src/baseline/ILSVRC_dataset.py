from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

valid_root = "/home/kousei/dataset/image_datasets/ILSVRC/2012/ILSVRC2012_img_val_for_ImageFolder"  # 検証データのフォルダ

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


if __name__ == "__main__":
    # 動作確認
    print("len", len(valid_dataset))
    img, label = valid_dataset[1]
    print("img = ", img)
    print("img_shape", img.shape)
    print("class(WordNet ID) = ", valid_dataset.classes[label])
    data_loader = data.DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
    )
    for imgs, labels in data_loader:
        print(imgs.size(), labels.size())
        break
