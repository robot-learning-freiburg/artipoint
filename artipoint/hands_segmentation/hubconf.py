"""
Torch Hub script for accessing te hand segmentation model outside the repo.
"""

##################################################
# Imports
##################################################

dependencies = ["torch", "pytorch_lightning"]

import torch
from model import HandSegModel
import gdown
import os


def hand_segmentor(pretrained=True, *args, **kwargs):
    """
    Hand segmentor based on a DeepLabV3 model with a ResNet50 encoder.
    DeeplabV3: https://arxiv.org/abs/1706.05587
    ResNet50: https://arxiv.org/abs/1512.03385
    """
    model = HandSegModel(*args, **kwargs)
    if pretrained:
        # check if the model is already downloaded
        checkpoint_path = kwargs.get(
            "checkpoint_path", "./checkpoints/hands_checkpoint.ckpt"
        )
        if not os.path.exists(checkpoint_path):
            _download_file_from_google_drive(
                "1w7dztGAsPHD_fl_Kv_a8qHL4eW92rlQg",
                checkpoint_path,
            )
        model = HandSegModel.load_from_checkpoint(
            map_location=torch.device("cuda"),
            *args,
            **kwargs,
        )
    return model


def _download_file_from_google_drive(id, destination):

    url = f"https://drive.google.com/uc?id={id}"
    path = os.path.dirname(destination)
    if not os.path.exists(path):
        os.makedirs(path)
    gdown.download(url, destination, quiet=False)
