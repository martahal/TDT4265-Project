from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
import logging
import os
import pathlib
import torch
from ssd.engine.inference import do_evaluation
from ssd.config.defaults import cfg
from ssd.utils.logger import setup_logger
from train import start_train
import numpy as np


def calculate_std_and_mean(config_path):
    config_file = config_path
    cfg.merge_from_file(config_file)
    cfg.freeze()
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger("SSD", output_dir)

    logger.info("Loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    max_iter = cfg.SOLVER.MAX_ITER
    train_loader = make_data_loader(cfg, is_train=True, max_iter=1000, start_iter=0)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, boxes, labels in train_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    mean = np.array(mean)
    mean = mean_to_rgb_range(mean)

    return std, mean

def calculate_priors(scales, image_size):
    """
        Calculates bounding box sizes and strides based on array of custom scales and image size.
    Args:
        scales: array of manually selected scales for determining prior size ranges
        image_size: the corresponding image size for the prior size ranges

    Returns:
        min_sizes: bounding box min sizes
        max_sizes: bounding box max sizes
        strides: distance between bounding boxes
    """
    sizes = [[int(round(scale * image_size[0])), int(round(scale * image_size[1]))]for scale in scales]
    min_sizes = sizes[0: -1]
    max_sizes = sizes[1:]
    new_feature_maps = calculate_feature_maps(image_size)
    new_feature_maps[-1] = [1,2]
    strides = [[int(np.ceil(image_size[0]/feature_map[0])),
                int(np.ceil(image_size[1]/feature_map[1]))]
               for feature_map in new_feature_maps]

    print('MIN_SIZES: ', min_sizes)
    print('MAX_SIZES: ', max_sizes)
    print('FEATURE_MAPS: ', new_feature_maps)
    print('STRIDES: ', strides)

def calculate_feature_maps(image_size,
                           standard_image_size = (300,300),
                           standard_fms=None):
    if standard_fms is None:
        standard_fms = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]
    new_fms = [[int(standard_fm[0] * image_size[0]/standard_image_size[0]),
                int(standard_fm[1] * image_size[1]/standard_image_size[1])]
               for standard_fm in standard_fms]

    return new_fms
def adust_prior_aspect_ratios(image_size, aspect_ratios, standard_image_size = (600,600)):
    std_ar =standard_image_size[1]/standard_image_size[0] # NOTE width/height
    convertion_factor = image_size[1]/image_size[0]
    new_ars = []
    for aspect_ratio in aspect_ratios:
        new_ratio=[]
        for ratio in aspect_ratio:
            new_ratio.append(round(ratio * convertion_factor,2))
        new_ars.append(new_ratio)
    print(new_ars)
    pass


def mean_to_rgb_range(mean):
    rgb_mean = []
    for value in mean:
        rgb_mean.append(round((value + 1) * 255 / 2, 3))
    return rgb_mean

def main():
    custom_scales = [0.0167, 0.083, 0.167, 0.25, 0.417, 0.667, 1.05]
    image_size = (255,450)
    calculate_priors(custom_scales, image_size)
    adust_prior_aspect_ratios(image_size, [[3], [2, 4], [2, 4], [2, 4], [4], [3]])
    #calculate_feature_maps(image_size)
    #std, mean = calculate_std_and_mean("configs/train_rdd2020_server_experimental.yaml")
    #print(std, mean)
    # Found that std and mean have the values listed below:
    #std = np.array([0.1868, 0.1933, 0.2099])
    #mean = np.array([-0.0183,  0.0138,  0.0443])

    #rgb_mean = mean_to_rgb_range(mean)

    #print(rgb_mean)

if __name__ == '__main__':
    main()