# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from collections import defaultdict
from typing import List

try:
    import detectron2
except ModuleNotFoundError:
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

try:
    import segment_anything
except ModuleNotFoundError:
    os.system('pip install git+https://github.com/facebookresearch/segment-anything.git')

# fmt: off
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from cat_seg import add_cat_seg_config
from demo.predictor import VisualizationDemo
import gradio as gr
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as fc

# constants
WINDOW_NAME = "CAT-Seg demo"


def setup_config(config_file: str, additional_options: List[str] = None):
    """

    Args:
        config_file (str): path to config file
        additional_options (List[str]): Modify config options using the command-line 'KEY VALUE' pairs

    Returns:
        the config object
    """
    # load config from file and command-line arguments
    if additional_options is None:
        additional_options = []
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(additional_options)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg


def save_masks(preds, text):
    preds = preds['sem_seg'].argmax(dim=0).cpu().numpy()  # C H W
    for i, t in enumerate(text):
        dir = f"mask_{t}.png"
        mask = preds == i
        cv2.imwrite(dir, mask * 255)


def save_prediction(input_img_path, output_img_path, text):
    cfg = setup_config('./configs/demo.yaml')
    demo = VisualizationDemo(cfg)
    img: np.ndarray = cv2.imread(input_img_path)

    predictions, vis_output = demo.run_on_image(img, text)

    vis_output.save(output_img_path)


def catseg_inference(catseg_visualizer: VisualizationDemo, image: np.ndarray, prompts: List[str]) -> dict:
    """
    Runs inference on the given image with a list
    Args:
        catseg_visualizer (VisualizationDemo): the catseg object
        image (np.ndarray): a BGR numpy ndarray represents an image
        prompts (List[str]): a list of prompts

    Returns:
        dict: a dictionary represents the {class: mask} inferences
    """
    text_prompt = ', '.join(prompts)
    predictions, _ = catseg_visualizer.run_on_image(image, text_prompt)
    aggregated_mask = predictions['sem_seg'].argmax(dim=0).cpu()
    class_mask_mapping = defaultdict(lambda: np.zeros_like(aggregated_mask, dtype=np.bool_))
    for i in range(len(prompts)):
        class_mask_mapping[prompts[i]][np.where(aggregated_mask == i)] = True

    return class_mask_mapping


if __name__ == "__main__":
    image = cv2.imread("/media/mafat/backup/omer_task/catseg_demo/CAT_Seg_demo/input/uav0000323_01173_v_0000006.jpg")
    prompts = ['car', 'roads', 'cars', 'tree', 'humans', 'pavement', 'building', 'trees']
    cfg = setup_config('./configs/demo.yaml')
    visualizer = VisualizationDemo(cfg)
    class_to_mask_mapping = catseg_inference(visualizer, image, prompts)
    plt.imshow(class_to_mask_mapping['humans'])

    # save_prediction("/media/mafat/backup/omer_task/catseg_demo/CAT_Seg_demo/input/uav0000323_01173_v_0000006.jpg", "uav0000323_01173_v_0000006.jpg",
    #                 "car, roads, cars, tree, humans, pavement, building, trees")
