# yolo v1
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BBOX = 2
GRID_SIZE = 7
CELL_SIZE = 64
IMAGE_SIZE = GRID_SIZE * CELL_SIZE
N_CLASSES = 20


class YoloReshape(nn.modules):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = tuple(target_shape)

    def forward(self, input):
        S = [self.target_shape[0], self.target_shape[1]]

        idx1 = S[0] * S[1] * N_CLASSES
        idx2 = idx1 + S[0] * S[1] * N_BBOX

        # class probabilities
        class_probs = input[:, :idx1].view(-1, S[0], S[1], N_CLASSES)
        class_probs = F.softmax(class_probs, dim=3)

        # confidence
        confs = input[:, idx1:idx2].view(-1, S[0], S[1], N_BBOX)
        confs = F.sigmoid(confs)

        # boxes
        boxes = input[:, idx2:].view(-1, S[0], S[1], N_BBOX * 4)
        boxes = F.sigmoid(boxes)

        outputs = torch.cat((class_probs, confs, boxes), 3)
        return outputs


# オプティマイザでweight decayを設定すること
Yolo = nn.Sequential(
    nn.Conv2d(3, 64, 7, stride=1, padding=3),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(2, stride=2, padding=1),
    nn.Conv2d(64, 192, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(2, stride=2, padding=1),
    nn.Conv2d(192, 128, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 256, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 256, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(2, stride=2, padding=1),
    nn.Conv2d(512, 256, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 256, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 256, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 256, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 512, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 1024, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(2, stride=2, padding=1),
    nn.Conv2d(1024, 512, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 1024, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 512, 1, stride=1, padding=0),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 1024, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
    nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
    nn.LeakyReLU(0.1),
    nn.Flatten(),
    nn.Linear(50176, 512),
    nn.Linear(512, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 1470),
    nn.Sigmoid(),
    YoloReshape((7, 7, 30)),
)
