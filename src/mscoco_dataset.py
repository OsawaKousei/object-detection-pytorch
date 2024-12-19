import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# 物体検出用のアノテーション情報
anno_path = (
    "/home/kousei/dataset/image_datasets/ms-coco/annotations/instances_val2017.json"
)

train_data_dir = "/home/kousei/dataset/image_datasets/ms-coco/train2017"
val_data_dir = "/home/kousei/dataset/image_datasets/ms-coco/val2017"


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
        return image, target

    def __len__(self):
        return len(self.ids)


train_dataset = CocoObjectDetection(
    root=train_data_dir,
    annFile=anno_path,
    transform=transforms.Compose([transforms.ToTensor()]),
)

val_dataset = CocoObjectDetection(
    root=val_data_dir,
    annFile=anno_path,
    transform=transforms.Compose([transforms.ToTensor()]),
)


if __name__ == "__main__":
    root = val_data_dir
    annFile = anno_path
    coco_dataset = CocoObjectDetection(
        root=root,
        annFile=annFile,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    print(f"Number of samples: {len(coco_dataset)}")

    # データセットからデータを取得
    img, target = coco_dataset[0]

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
