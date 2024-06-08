import torch
from torch.utils.data import Dataset


class CustomTensorData(Dataset):
    def __init__(self, tensor, transform=None) -> None:
        super().__init__()
        self.tensor = tensor
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        image = self.tensor[0][index]
        label = self.tensor[1][index]
        bbox = self.tensor[2][index]

        image = image.premute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, label, bbox

    def __len__(self) -> int:
        return self.tensor[0].size(0)
