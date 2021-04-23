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


def calculate_std_and_mean(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, boxes, labels in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return std, mean

def mean_to_rgb_range(mean):
    rgb_mean = []
    for value in mean:
        rgb_mean.append(round((value + 1) * 255 / 2, 3))
    return rgb_mean

def main():
    #config_file = "configs/train_rdd2020_server_experimental_contrast.yaml"
    #cfg.merge_from_file(config_file)
    #cfg.freeze()
    #output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    #output_dir.mkdir(exist_ok=True, parents=True)
#
    #logger = setup_logger("SSD", output_dir)
#
    #logger.info("Loaded configuration file {}".format(config_file))
    #with open(config_file, "r") as cf:
    #    config_str = "\n" + cf.read()
    #    logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))
#
    #max_iter = cfg.SOLVER.MAX_ITER
    #train_loader = make_data_loader(cfg, is_train=True, max_iter=1000, start_iter=0)

    #std, mean = calculate_std_and_mean(train_loader)
    # Found that std and mean have the values listed below:
    # std = np.array([0.1868, 0.1933, 0.2099])
    # mean = np.array([-0.0183,  0.0138,  0.0443])
    std =[0.2730, 0.2721, 0.2664]
    mean = np.array([0.0126, 0.0435, 0.0930])
    print(std, mean)
    
    rgb_mean = mean_to_rgb_range(mean)
    
   
    print(rgb_mean)

if __name__ == '__main__':
    main()