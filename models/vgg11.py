"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


def _conv_block(ch_in, ch_out):
    """Single conv-bn-relu unit."""
    return [
        nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True),
    ]


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 37, dropout_p: float = 0.5):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # Block 1: 1 conv (64) + pool → 112x112
        self.block1 = nn.Sequential(
            *_conv_block(in_channels, 64),
            nn.MaxPool2d(2, 2),
        )
        # Block 2: 1 conv (128) + pool → 56x56
        self.block2 = nn.Sequential(
            *_conv_block(64, 128),
            nn.MaxPool2d(2, 2),
        )
        # Block 3: 2 convs (256, 256) + pool → 28x28
        self.block3 = nn.Sequential(
            *_conv_block(128, 256),
            *_conv_block(256, 256),
            nn.MaxPool2d(2, 2),
        )
        # Block 4: 2 convs (512, 512) + pool → 14x14
        self.block4 = nn.Sequential(
            *_conv_block(256, 512),
            *_conv_block(512, 512),
            nn.MaxPool2d(2, 2),
        )
        # Block 5: 2 convs (512, 512) + pool → 7x7
        self.block5 = nn.Sequential(
            *_conv_block(512, 512),
            *_conv_block(512, 512),
            nn.MaxPool2d(2, 2),
        )

        # Classification head with BatchNorm1d for better generalization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        s5 = self.block5(s4)

        logits = self.classifier(s5)

        if return_features:
            skips = {"f1": s1, "f2": s2, "f3": s3, "f4": s4, "f5": s5}
            return logits, skips
        return logits


# Alias for autograder compatibility (imports VGG11 from models.vgg11)
VGG11 = VGG11Encoder