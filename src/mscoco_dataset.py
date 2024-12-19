import matplotlib.pyplot as plt
import torch
from PIL import Image
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

        # アノテーションが存在する画像IDを取得
        valid_image_ids = {
            ann["image_id"] for ann in self.coco.loadAnns(self.coco.getAnnIds())
        }
        # アノテーションが存在する画像のみをフィルタリング
        self.ids = [
            img["id"] for img in self.coco.imgs.values() if img["id"] in valid_image_ids
        ]
        self.images = [self.coco.imgs[img_id] for img_id in self.ids]

        # データセットの長さを更新
        self.length = len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        img_info = self.coco.imgs[image_id]
        img_path = self.coco.loadImgs(image_id)[0]["file_name"]
        img_path = f"{self.root}/{img_path}"

        # 画像の読み込み
        image = Image.open(img_path).convert("RGB")
        # 画像のリサイズ
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # アノテーションの取得
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # アノテーションが存在しない場合はエラーを投げる
        if len(anns) == 0:
            raise IndexError(f"No annotations found for image_id {image_id}")

        # ラベルとボックスの取得
        labels = torch.tensor([ann["category_id"] for ann in anns], dtype=torch.long)
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32)

        # バウンディングボックスが1つの場合でも2次元にする
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)

        # 画像のサイズを取得
        w, h = img_info["width"], img_info["height"]

        # バウンディングボックスを正規化
        boxes[:, [0, 2]] /= w  # x_min, x_max
        boxes[:, [1, 3]] /= h  # y_min, y_max

        # バウンディングボックスをIMG_SIZEを基準にリサイズ
        boxes *= IMG_SIZE

        target = {
            "labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([image_id], dtype=torch.long),
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return self.length


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


def od_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])

    return torch.stack(imgs, dim=0), targets


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
