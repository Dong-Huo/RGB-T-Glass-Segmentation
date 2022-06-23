# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import random
import sys
from typing import Iterable

import cv2
import torch
import numpy as np
import util.misc as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, batch_size, max_norm: float = 0, is_rgbt=True):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    iter_num = math.ceil(len(data_loader.dataset) / batch_size)

    for i, (rgb, temperature, targets, img_path) in enumerate(data_loader):
        rgb = rgb.to(device)
        temperature = temperature.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        max_h, max_w = target_sizes.max(0)[0].tolist()

        final_size = (max_h, max_w)

        # change True to False for RGB only
        outputs = model(rgb, temperature, final_size, is_rgbt)
        loss_dict = criterion(outputs, targets)
        loss = sum([loss_dict[k] for k in loss_dict.keys()])

        loss_name_list = [k for k in loss_dict.keys()]

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        for j in range(len(loss_name_list)):
            writer.add_scalar('training/' + loss_name_list[j], loss_dict[loss_name_list[j]].item(),
                              epoch * iter_num + i)

        print('Epoch: [{}], iteration: [{}], loss: [{}]'.format(epoch, i, loss.item()))

        del loss_dict
    torch.cuda.empty_cache()


def iou(pred, target):
    pred = pred > 0.5
    pred = pred.astype(np.bool)
    target = target.astype(np.bool)

    if np.isclose(np.sum(pred), 0) and np.isclose(np.sum(target), 0):
        return 1
    else:
        return np.sum((pred & target)) / np.sum((pred | target), dtype=np.float32)


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, is_rgbt=True):
    model.eval()
    criterion.eval()

    rgbt_iou_list = []
    rgbt_mae_list = []

    for rgb, temperature, targets, img_path in data_loader:

        if "noglass" in img_path[0]:
            continue

        rgb = rgb.to(device)
        temperature = temperature.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        max_h, max_w = target_sizes.max(0)[0].tolist()

        final_size = (max_h, max_w)

        # change True to False for RGB only
        outputs = model(rgb, temperature, final_size, is_rgbt)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        orig_masks = [t["orig_masks"] for t in targets]

        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        results = postprocessors['mask'](outputs, orig_target_sizes, target_sizes)

        for i in range(len(orig_masks)):
            result = results[i].squeeze().cpu().numpy()
            orig_mask = orig_masks[i].squeeze().cpu().numpy()

            rgbt_iou_list.append(iou(result, orig_mask))
            rgbt_mae_list.append(np.mean(np.abs(np.float32(result) - np.float32(orig_mask))))

            scene_name = img_path[i].split("/")[-3]
            image_name = img_path[i].split("/")[-2]

            print(os.path.join("results", scene_name))
            os.makedirs(os.path.join("results", scene_name), exist_ok=True)
            cv2.imwrite(os.path.join("results", scene_name, image_name + ".png"), result * 255)

        del outputs, orig_target_sizes, orig_masks, target_sizes, results
    torch.cuda.empty_cache()

    rgb_iou_list = []
    rgb_mae_list = []

    for rgb, temperature, targets, img_path in data_loader:

        if "noglass" not in img_path[0]:
            continue

        rgb = rgb.to(device)
        temperature = temperature.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        max_h, max_w = target_sizes.max(0)[0].tolist()

        final_size = (max_h, max_w)

        # change True to False for RGB only
        outputs = model(rgb, temperature, final_size, is_rgbt)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        orig_masks = [t["orig_masks"] for t in targets]

        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        results = postprocessors['mask'](outputs, orig_target_sizes, target_sizes)

        for i in range(len(orig_masks)):
            result = results[i].squeeze().cpu().numpy()
            orig_mask = orig_masks[i].squeeze().cpu().numpy()

            rgb_iou_list.append(iou(result, orig_mask))
            rgb_mae_list.append(np.mean(np.abs(np.float32(result) - np.float32(orig_mask))))

            scene_name = img_path[i].split("/")[-3]
            image_name = img_path[i].split("/")[-2]

            print(os.path.join("results", scene_name))
            os.makedirs(os.path.join("results", scene_name), exist_ok=True)
            cv2.imwrite(os.path.join("results", scene_name, image_name + ".png"), result * 255)

        del outputs, orig_target_sizes, orig_masks, target_sizes, results
    torch.cuda.empty_cache()

    return sum(rgbt_iou_list) / len(rgbt_iou_list), sum(rgbt_mae_list) / len(rgbt_mae_list), sum(rgb_iou_list) / len(
        rgb_iou_list), sum(rgb_mae_list) / len(rgb_mae_list)
