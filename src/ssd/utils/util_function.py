import torch


def od_collate_fn(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        x_min, y_min, x_max, y_max = sample[1]["boxes"].T
        labels = sample[1]["labels"]
        target = torch.stack([labels, x_min, y_min, x_max, y_max], dim=1)
        targets.append(target)

    return torch.stack(imgs, dim=0), targets
