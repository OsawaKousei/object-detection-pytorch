import os
import urllib.request

# フォルダ「weights」が存在しない場合は作成する
weights_dir = "src/ssd/weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)


# 学習済みのSSD用のVGGのパラメータをフォルダ「weights」にダウンロード
# MIT License
# Copyright (c) 2017 Max deGroot, Ellis Brown
# https://github.com/amdegroot/ssd.pytorch

url = "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
target_path = os.path.join(weights_dir, "vgg16_reducedfc.pth")

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)

# 学習済みのSSD300モデルをフォルダ「weights」にダウンロード
# MIT License
# Copyright (c) 2017 Max deGroot, Ellis Brown
# https://github.com/amdegroot/ssd.pytorch

url = "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth"
target_path = os.path.join(weights_dir, "ssd300_mAP_77.43_v2.pth")

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)
