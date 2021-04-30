import logging
import glob
import os
import gzip
import matplotlib.pyplot as plt
from typing import Tuple

import numpy as np
import albumentations as A
from torchvision.transforms import ToTensor, Compose, RandomRotation
import torchvision.transforms.functional as F
import torch
from torch import Tensor


class PairwiseCompose(Compose):

    def __call__(self, input):
        if isinstance(input, (list, tuple)):
            return self.apply_sequence(input)
        else:
            return self.apply_img(input)

    def apply_img(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def apply_sequence(self, seq):
        img, mask = seq[0], seq[1]
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class PairwiseRandomRotation(RandomRotation):

    def __call__(self, img, mask):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        if isinstance(mask, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(mask)
            else:
                fill = [float(f) for f in fill]

        angle = self.get_params(self.degrees)

        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        mask = F.rotate(mask, angle, self.resample, self.expand, self.center)

        return img, mask


def get_transformed_image_and_target(
        image: np.ndarray,
        mask: np.ndarray,
        show: bool = True
) -> Tuple[np.ndarray, np.ndarray]:

    transform = PairwiseCompose([
        PairwiseRandomRotation(degrees=360)
    ])

    transformed_image, transformed_mask = transform(input=[torch.from_numpy(image), torch.from_numpy(mask)])
    if show:
        image_idx = 71

        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(image[image_idx, :, :], cmap="Greys")
        ax[0].set_title("image")
        ax[1].imshow(transformed_image[image_idx, :, :], cmap="Greys")
        ax[1].set_title("transformed_image")
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(mask[image_idx, :, :], cmap="Greys")
        ax[0].set_title("mask")
        ax[1].imshow(transformed_mask[image_idx, :, :], cmap="Greys")
        ax[1].set_title("transformed_mask")
        plt.show()

    return transformed_image.numpy(), transformed_mask.numpy()


def save_augmented_dataset(
        path: str = "data",
        n_batch=4
):
    """
    Save dataset of images in the folder path after applying basic data augmentation.

    Args:
        path (str): The path to the data files
        n_batch (int): The number of files to load (max value is 4)

    Return:
        nothing
    """
    logging.info(f"\n{'-' * 25}\nGenerating AUG data files\n{'-' * 25}")
    for batch in range(1, n_batch + 1):
        phantom_file = glob.glob(os.path.join(path, "Phantom_batch{}.npy*".format(batch)))[0]
        logging.info(f"Loading file {phantom_file}")
        try:
            f = gzip.GzipFile(phantom_file, "r")
            phantom = np.load(f)
            f.close()
        except:
            phantom = np.load(phantom_file)

        fbp_file = glob.glob(os.path.join(path, "FBP128_batch{}.npy*".format(batch)))[0]
        logging.info(f"Loading file {fbp_file}")
        try:
            f = gzip.GzipFile(fbp_file, "r")
            fbp = np.load(f)
            f.close()
        except:
            fbp = np.load(fbp_file)

        transformed_image, transformed_mask = get_transformed_image_and_target(
            image=fbp,
            mask=phantom
        )

        aug_file = "./{}/AUG_batch{}.npy".format(path, batch)
        np.save(aug_file, transformed_image)
        logging.info(f"Data augmentation dataset for batch {batch} saved to file {aug_file}")

        aug_target_file = "./{}/AUG_target_batch{}.npy".format(path, batch)
        np.save(aug_target_file, transformed_mask)
        logging.info(f"Data augmentation dataset for batch {batch} saved to file {aug_target_file}")
