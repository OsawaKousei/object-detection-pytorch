import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

train_anno_path = (
    "/home/kousei/dataset/image_datasets/ms-coco/annotations/instances_train2017.json"
)
val_anno_path = (
    "/home/kousei/dataset/image_datasets/ms-coco/annotations/instances_val2017.json"
)

train_data_dir = "/home/kousei/dataset/image_datasets/ms-coco/train2017"
val_data_dir = "/home/kousei/dataset/image_datasets/ms-coco/val2017"

IMG_SIZE = 300


class CocoObjectDetection(datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CocoObjectDetection, self).__init__(
            root, annFile, transform, target_transform
        )

    def __getitem__(self, index):
        image, target = super(CocoObjectDetection, self).__getitem__(index)
        target = {
            "image_id": target[0]["image_id"],
            "labels": torch.tensor([obj["category_id"] for obj in target]),
            "boxes": torch.tensor([obj["bbox"] for obj in target]),
        }
        # bboxの値を画像のサイズで割って正規化
        _, w, h = image.size()
        target["boxes"][:, [1, 3]] /= w
        target["boxes"][:, [0, 2]] /= h

        # bboxの値をIMG_SIZEを基準にリサイズ
        target["boxes"] *= IMG_SIZE

        # 画像をリサイズ
        image = transforms.functional.resize(image, (IMG_SIZE, IMG_SIZE))

        return image, target

    def __len__(self):
        return len(self.ids)


train_dataset = CocoObjectDetection(
    root=train_data_dir,
    annFile=train_anno_path,
    transform=transforms.Compose([transforms.ToTensor()]),
    target_transform=None,
)

val_dataset = CocoObjectDetection(
    root=val_data_dir,
    annFile=val_anno_path,
    transform=transforms.Compose([transforms.ToTensor()]),
    target_transform=None,
)


if __name__ == "__main__":
    print(f"Number of samples: {len(train_dataset)}")

    # データセットからデータを取得
    img, target = train_dataset[0]

    print(f"Image size: {img.size()}")

    # 画像の表示
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("coco_image.png")

    # アノテーション情報の表示
    print(target)

    # アノテーション情報の可視化
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img.permute(1, 2, 0))
    for box in target["boxes"]:
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    plt.savefig("coco_image_with_bbox.png")
