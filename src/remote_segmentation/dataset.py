# -*- coding: utf-8 -*-

import os
import torch

from typing import Union, Tuple
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

__all__ = ["FloodSegmentationDataset"]


class FloodSegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        size: Union[int, Tuple[int, int]] = 224,
        num_label_pixels: int = 1000,
        mask_value: int = 255,
        augment: bool = False,
    ):
        super(FloodSegmentationDataset, self).__init__()

        split = split.lower()

        assert split in [
            "train",
            "val",
            "test",
        ], "split must be 'train' or 'val' or 'test'"

        if isinstance(size, int):
            size = (size, size)

        self.size = size

        self.split = split
        self.num_label_pixels = num_label_pixels
        self.mask_value = mask_value

        self.masked_H = torch.randint(
            low=0, high=self.size[0], size=(num_label_pixels,)
        )
        self.masked_W = torch.randint(
            low=0, high=self.size[1], size=(num_label_pixels,)
        )

        if augment:
            augmentation = T.AutoAugment()
            data_transform_list = [
                augmentation,
                T.Resize(self.size, interpolation=Image.NEAREST),
                T.ToTensor(),
            ]
        else:
            data_transform_list = [
                T.Resize(self.size, interpolation=Image.NEAREST),
                T.ToTensor(),
            ]

        data_transform = T.Compose(data_transform_list)

        mask_transform = T.Compose(
            [
                T.Resize(self.size, interpolation=Image.NEAREST),
                T.PILToTensor(),
                T.Lambda(lambda img: img.to(torch.int64)),
            ]
        )

        self.data_dir = data_dir

        self.data_transform = data_transform
        self.mask_transform = mask_transform

        img_paths = os.path.join(data_dir, self.split, f"{split}-org-img")
        label_paths = os.path.join(data_dir, self.split, f"{split}-label-img")

        img_files = [os.path.join(img_paths, f) for f in os.listdir(img_paths)]
        label_files = [os.path.join(label_paths, f) for f in os.listdir(label_paths)]

        self.files = img_files
        self.labels = label_files

        with open(f"{data_dir}/DATASET-VERSION-NOTE-v1.0.txt", "r") as f:
            lines = f.readlines()

        self.num_classes = int(lines[-3].strip().split(" ")[3])

    def __len__(self) -> int:
        return len(self.files)

    def load_image(self, path: str, mask: bool = False) -> torch.Tensor:
        transform = self.mask_transform if mask else self.data_transform
        # cv2_transform = T.Compose(
        #     [T.ToTensor(), T.Resize(self.size, interpolation=Image.NEAREST)]
        # )
        # pil_transform = T.Compose(
        #     [T.Resize(self.size, interpolation=Image.NEAREST), T.ToTensor()]
        # )

        image = Image.open(path)
        # image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        return transform(image)

    def simulate_point_labels(
        self,
        segmentation_mask: torch.Tensor,
    ) -> torch.Tensor:
        sampled_mask = torch.ones_like(segmentation_mask) * self.mask_value
        sampled_mask[:, self.masked_H, self.masked_W] = segmentation_mask[
            :, self.masked_H, self.masked_W
        ]

        return sampled_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.load_image(self.files[idx], mask=False)
        segmentation_mask = self.load_image(self.labels[idx], mask=True)
        segmentation_mask = self.simulate_point_labels(
            segmentation_mask=segmentation_mask
        )

        return (image, segmentation_mask)

    # def __getitem__(self, idx: int):
    #     return torch.randn(3, 224, 224), torch.randint(0, self.num_classes, (1, 224, 224))

    def to_dataloader(
        self,
        num_workers: int = 0,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=self.split == "train",
        )
        return dataloader


# if __name__ == "__main__":
#     data_dir = "/mnt/c/Users/Harkhymadhe/AIWorkspace/meriti/dataset/FloodNet/FloodNet-Supervised_v1.0"
#     dataset = FloodSegmentationDataset(data_dir=data_dir, num_label_pixels=224*224)
#     loss = PartialCrossEntropyLoss(num_classes=10)

#     index = 129
#     im, target = dataset[index]

#     print(target.shape)
#     print(target)
#     print(torch.unique(target))

#     # m_pil = T.functional.to_pil_image(m.to(torch.uint8))

#     # m_pil.show()


#     # mask_path = dataset.labels[index]
#     # print(mask_path)
#     # img_pil = Image.open(mask_path).convert("RGB")
#     # img_pil.show()
#     # img = T.PILToTensor()(img_pil)
#     # print(img)
#     # print(torch.unique(img))

#     pred = torch.randn(size = (1, 10, 224, 224))

#     loss_ = loss(pred, target.unsqueeze(0))

#     # mask = np.array(Image.open(mask_path))
#     # print(np.unique(mask))
