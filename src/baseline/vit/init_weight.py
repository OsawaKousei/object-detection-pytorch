# # python3 -m src.baseline.vit.init_weight

# import os

# import numpy as np
# import torch
# import torch.nn as nn

# from src.baseline.vit.vit_model import Vit

# param_dict: dict[str, torch.Tensor] = {}

# root_dir = "/home/kousei/dataset/pretrained/vit_B-16_ImageNet21k"  # 重みが保存されているディレクトリのパスを指定

# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for filename in filenames:
#         if filename.endswith(".npy"):
#             file_path = os.path.join(dirpath, filename)
#             # パラメータ名を生成
#             relative_path = os.path.relpath(file_path, root_dir)
#             param_name = relative_path.replace(os.sep, "-").replace(".npy", "")
#             # npyファイルをロードしてTensorに変換
#             np_array = np.load(file_path)
#             tensor = torch.from_numpy(np_array)
#             # 辞書に格納
#             param_dict[param_name] = tensor

# print(param_dict.keys())


# def init_param(
#     param: torch.nn.Parameter,
#     param_name: str,
#     base_key: str,
# ) -> None:
#     if "weight" in param_name:
#         param.data = param_dict[base_key + "-kernel"]
#     elif "bias" in param_name:
#         param.data = param_dict[base_key + "-bias"]
#     else:
#         raise ValueError(f"Unexpected param_name: {param_name}")


# def init_weight(model: nn.Module) -> nn.Module:
#     # モデルのパラメータを取得
#     for name, param in model.named_parameters():
#         # param_dictのキーと対応するようにパラメータ名を変換
#         if "cls_token" in name:
#             param.data = param_dict["cls"]
#         elif "pos_emb" in name:
#             key = "Transformer-posembed_input-pos_embedding"
#         elif name == "input_layer.patch_emb_layer.weight":
#             key = "embedding-kernel"
#         elif name == "input_layer.patch_emb_layer.bias":
#             key = "embedding-bias"
#         else:
#             # エンコーダブロックのパラメータ名を変換
#             key = name.replace("encorder.", "Transformer-encoderblock_")
#             key = key.replace(".ln1.weight", "-LayerNorm_0-scale")
#             key = key.replace(".ln1.bias", "-LayerNorm_0-bias")
#             key = key.replace(".ln2.weight", "-LayerNorm_2-scale")
#             key = key.replace(".ln2.bias", "-LayerNorm_2-bias")
#             key = key.replace(
#                 ".msa.w_q.weight", "-MultiHeadDotProductAttention_1-query-kernel"
#             )
#             key = key.replace(
#                 ".msa.w_k.weight", "-MultiHeadDotProductAttention_1-key-kernel"
#             )
#             key = key.replace(
#                 ".msa.w_v.weight", "-MultiHeadDotProductAttention_1-value-kernel"
#             )
#             key = key.replace(
#                 ".msa.w_o.0.weight", "-MultiHeadDotProductAttention_1-out-kernel"
#             )
#             key = key.replace(
#                 ".msa.w_o.0.bias", "-MultiHeadDotProductAttention_1-out-bias"
#             )
#             key = key.replace(".mlp.0.weight", "-MlpBlock_3-Dense_0-kernel")
#             key = key.replace(".mlp.0.bias", "-MlpBlock_3-Dense_0-bias")
#             key = key.replace(".mlp.3.weight", "-MlpBlock_3-Dense_1-kernel")
#             key = key.replace(".mlp.3.bias", "-MlpBlock_3-Dense_1-bias")
#             # インデックスを調整
#             key = key.replace(".", "-")
#         # 重みを初期化
#         if key in param_dict:
#             param.data = param_dict[key]
#         else:
#             print(f"{key} は param_dict に存在しません。")
#     return model


# if __name__ == "__main__":
#     model = Vit()

#     for name, param in model.named_parameters():
#         print(name)
