# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as T

from remote_segmentation.dataset import FloodSegmentationDataset

__all__ = [
    "RandomHistogramMatchingTransformerV1",
    "RandomHistogramMatchingTransformerV2",
    "RandomHistogramMatchingTransformer",
]


class RandomHistogramMatchingTransformerV1(torch.nn.Module):
    def __init__(self, target_dataset):
        super(RandomHistogramMatchingTransformer, self).__init__()

        self.target_dataset = target_dataset

    def _generate_cmf(self, image, channel=0):
        flat_img = image[:, channel, :, :].flatten()
        flat_img = torch.sort(flat_img, dim=0, descending=False)[0]
        flat_prop = torch.bincount(flat_img)
        flat_cmf = flat_prop.cumsum(dim=0) / flat_prop.max()
        return flat_cmf

    def generate_cmf(self, image):
        cmf = torch.stack(
            [self._generate_cmf(image, channel=c) for c in range(3)], dim=0
        )
        return cmf

    def inverse_cmf(self, cmf, proportion):
        return (cmf - proportion).argmin(dim=0).min()

    def match_histograms(self, source_cmf, target_cmf, image, channel=0):
        if len(source_cmf.shape) > 3:
            source_cmf = source_cmf[:, channel, :, :]
        else:
            source_cmf = source_cmf[channel]

        if len(target_cmf.shape) > 3:
            target_cmf = target_cmf[:, channel, :, :]
        else:
            target_cmf = target_cmf[channel]

        if len(image.shape) > 3:
            image = image[:, channel, :, :]
        else:
            image = image[channel]

        props = source_cmf[image.flatten()]

        return self.inverse_cmf(target_cmf, props)

    def sample_target_images(self, num_images):
        indices = torch.randint(
            low=0, high=len(self.target_dataset), size=(num_images,)
        )
        indices = indices.numpy().tolist()
        target_image = self.target_dataset[indices]
        return target_image

    @torch.inference_mode()
    def forward(self, source_image, target_image=None):
        source_cmf = self.generate_cmf(source_image)

        if target_image is None:
            target_image = self.sample_target_images(num_images=len(source_image))
        else:
            if len(source_image) > len(target_image):
                target_image = torch.cat(
                    [
                        target_image
                        for _ in range(len(source_image) - len(target_image))
                    ],
                    dim=0,
                )
                target_image = target_image[torch.randperm(n=len(target_image))]
                target_image = target_image[: len(source_image)]

            else:
                target_image = target_image[: len(source_image)]

        target_cmf = self.generate_cmf(target_image)

        shape = list(source_image.shape)
        del shape[1]

        matched_histograms = [
            self.match_histograms(
                source_cmf, target_cmf, target_image, channel=channel
            ).view(shape)
            for channel in range(3)
        ]
        return torch.stack(matched_histograms, dim=1)

    def to_image(self, image):
        image = image.cpu()
        images = [T.ToPILImage()(image[i]) for i in range(len(image))]

        for i, image_ in enumerate(images, 1):
            image.save(f"image_{i}.png")


class RandomHistogramMatchingTransformerV2(nn.Module):
    def __init__(self, target_dataset):
        super(RandomHistogramMatchingTransformer, self).__init__()
        self.target_dataset = target_dataset

    def sample_target_images(self, num_images: torch.Tensor) -> torch.Tensor:
        # Efficiently sample a batch from the target dataset
        indices = torch.randint(
            low=0, high=len(self.target_dataset), size=(num_images,)
        )
        # Assuming target_dataset returns a tensor or can be indexed by a list
        target_images = torch.stack([self.target_dataset[i][0] for i in indices])
        return target_images

    @torch.inference_mode()
    def forward(self, source_images: torch.Tensor, target_images: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        source_images: Tensor of shape [B, C, H, W]
        """
        B, C, H, W = source_images.shape

        if target_images is None:
            target_images = self.sample_target_images(num_images=B).to(
                source_images.device
            )

        # Ensure target_images matches source batch size
        if target_images.shape[0] != B:
            # Simple tile/truncate logic to match batch sizes
            target_images = target_images.repeat(
                (B // target_images.shape[0]) + 1, 1, 1, 1
            )[:B]

        # 1. Flatten for batch processing: [B, C, N]
        source_flat = source_images.view(B, C, -1)
        target_flat = target_images.view(B, C, -1)

        # 2. Sort both to get the distribution "templates"
        # We need the values in order to map percentiles
        source_sorted, source_indices = torch.sort(source_flat, dim=-1)
        target_sorted, _ = torch.sort(target_flat, dim=-1)

        # 3. Use the sorted target as a look-up table.
        # Since source_sorted is the sorted version of source_flat,
        # we can directly map the ranks.
        # 'target_sorted' now contains the values we WANT at each rank.

        # To handle the mapping back to the original pixel positions:
        # We create an empty tensor and scatter the target values into source positions
        matched_flat = torch.zeros_like(source_flat)
        matched_flat.scatter_(dim=-1, index=source_indices, src=target_sorted)

        return matched_flat.view(B, C, H, W)

    def save_images(self, tensor_batch: torch.Tensor, prefix: str = "image"):
        # Helper to save results
        for i, img_tensor in enumerate(tensor_batch):
            img = T.ToPILImage()(img_tensor.cpu())
            img.save(f"{prefix}_{i}.png")


RandomHistogramMatchingTransformer = RandomHistogramMatchingTransformerV1

if __name__ == "__main__":
    import os
    from PIL import Image
    from torchvision import transforms as T

    num_classes = 10
    reduce = True

    transform = T.Compose([T.PILToTensor(), T.Resize((4000, 3000))])

    base_dir = "../dataset/FloodNet/FloodNet-Supervised_v1.0/train"

    target_dataset = FloodSegmentationDataset(split="test")

    matcher = RandomHistogramMatchingTransformer(target_dataset=target_dataset)

    images = os.listdir(os.path.join(base_dir, "train-org-img"))[:5]
    images = [os.path.join(base_dir, "train-org-img", f) for f in images]

    labels = os.listdir(os.path.join(base_dir, "train-org-img"))[5:10]
    labels = [os.path.join(base_dir, "train-org-img", f) for f in labels]

    images = torch.stack([transform(Image.open(f)) for f in images], dim=0).to(
        torch.int64
    )
    labels = torch.stack([transform(Image.open(f)) for f in labels], dim=0).to(
        torch.int64
    )

    results = matcher(images, labels)
    matcher.to_image(results)
