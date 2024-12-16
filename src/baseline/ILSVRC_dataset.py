from torchvision import transforms
from torchvision.datasets import ImageFolder

valid_root = "/home/kousei/dataset/image_datasets/ILSVRC/2012/ILSVRC2012_img_val_for_ImageFolder"  # 検証データのフォルダ

valid_transform = transforms.Compose(
    [
        transforms.Resize(224),  # 1辺が224ピクセルの正方形に変換
        transforms.ToTensor(),  # Tensor行列に変換
    ]
)
valid_dataset = ImageFolder(
    root=valid_root, transform=valid_transform  # 画像が保存されているフォルダのパス
)  # Tensorへの変換
