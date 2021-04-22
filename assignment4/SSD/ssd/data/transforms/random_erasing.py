from __future__ import absolute_import

from torchvision.transforms import *
from transforms import ConvertFromInts, ToAbsoluteCoords
from contrast import Contrast
from PIL import Image
import random
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2


# Copied from https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
# Modifications have been applied to make it work with this codebase


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=1, sl=0.02, sh=0.08, r1=0.3, mean=[123.675, 116.280, 103.530]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, boxes=None, labels=None):

        if random.uniform(0, 1) > self.probability:
            return img, boxes, labels

        print(img.shape[0], img.shape[1])
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area

            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[0] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - w)
                y1 = random.randint(0, img.shape[0] - h)
                if img.shape[2] == 3:
                    img[y1:y1 + w, x1:x1 + h, 0] = self.mean[0]
                    img[y1:y1 + w, x1:x1 + h, 1] = self.mean[1]
                    img[y1:y1 + w, x1:x1 + h, 2] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img, boxes, labels

        return img, boxes, labels


def visualize():
    transform = [
        Contrast(),
        ConvertFromInts(),
        ToAbsoluteCoords(),
        RandomErasing(),
    ]
    # img = cv2.imread('../../../datasets/RDD2020_filtered/JPEGImages/Japan_007344.jpg', 1)
    img = cv2.imread('Image.png', 1)
    boxes = None
    labels = None
    for t in transform:
        img, boxes, labels = t(img, boxes, labels)

    cv2.imwrite('erasing.png', img)

visualize()