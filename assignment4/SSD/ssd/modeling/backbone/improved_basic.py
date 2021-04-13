import torch
from torch import nn
from collections import OrderedDict


class ImprovedBasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                ),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                ),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=output_channels[0],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(num_features=output_channels[0]),
            )),
            ('layer2', nn.Sequential(
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[0],
                    out_channels=output_channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[0],
                    out_channels=output_channels[1],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(num_features=output_channels[1]),
            )),
            ('layer3', nn.Sequential(
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[1],
                    out_channels=output_channels[1],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[1],
                    out_channels=output_channels[2],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(num_features=output_channels[2]),
            )),
            ('layer4',nn.Sequential(
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[2],
                    out_channels=output_channels[2],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[2],
                    out_channels=output_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(num_features=output_channels[3]),
            )),
            ('layer5',nn.Sequential(
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[3],
                    out_channels=output_channels[3],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[3],
                    out_channels=output_channels[4],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(num_features=output_channels[4]),
            )),
            ('layer6', nn.Sequential(
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[4],
                    out_channels=output_channels[4],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ELU(),
                nn.Conv2d(
                    in_channels=output_channels[4],
                    out_channels=output_channels[5],
                    kernel_size=3,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(num_features=output_channels[5]),
            ))
        ]))

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        l1_feature = self.feature_extractor.layer1(x)
        l2_feature = self.feature_extractor.layer2(l1_feature)
        l3_feature = self.feature_extractor.layer3(l2_feature)
        l4_feature = self.feature_extractor.layer4(l3_feature)
        l5_feature = self.feature_extractor.layer5(l4_feature)
        l6_feature = self.feature_extractor.layer6(l5_feature)
        out_features = [l1_feature,
                        l2_feature,
                        l3_feature,
                        l4_feature,
                        l5_feature,
                        l6_feature]

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

