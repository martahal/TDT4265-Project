import torchvision
import torch
from torch import nn
from collections import OrderedDict
from torchvision.models.resnet import resnet18


class ResNet18(torch.nn.Module):
    """
    A custom resnet backbone for SSD
    """

    def __init__(self, cfg):
        super().__init__()
        self.model = resnet18(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, progress=False)
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.image_channels = image_channels
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1, self.model.layer2)),
            ('layer2', self.model.layer3),
            ('layer3', self.model.layer4),
            ('layer4',  nn.Sequential(
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
            ('layer5', nn.Sequential(
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
                nn.BatchNorm2d(num_features=output_channels[4]))
             ),
            ('layer6', nn.Sequential( # For images with aspect ratio similar to 1080x1920 images
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
                    kernel_size=2,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(num_features=output_channels[5]),
            ))
            #('layer6', nn.Sequential(# For images with square aspect ratio 
            #    nn.ELU(),
            #    nn.Conv2d(
            #        in_channels=output_channels[4],
            #        out_channels=output_channels[4],
            #        kernel_size=3,
            #        stride=1,
            #        padding=1
            #    ),
            #    nn.ELU(),
            #    nn.Conv2d(
            #        in_channels=output_channels[4],
            #        out_channels=output_channels[5],
            #        kernel_size=3,
            #        stride=2,
            #        padding=0
            #    ),
            #    nn.BatchNorm2d(num_features=output_channels[5]),
            #))
        ]))

    def forward(self, x):
        """
        The forward function should output features with shape:
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
        # x = self.model(x)
 
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

