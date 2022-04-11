# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import datasets.transforms as T


class RGBT:
    def __init__(self, img_folder, transforms=None, return_masks=True):

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        # self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # # sanity check
        # if "annotations" in self.coco:
        #     for img, ann in zip(self.coco['images'], self.coco['annotations']):
        #         assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.transforms = transforms
        self.return_masks = return_masks

        self.image_path_list = []

        scene_list = os.listdir(img_folder)

        for scene_name in scene_list:
            if "." in scene_name:
                continue

            image_list = os.listdir(os.path.join(img_folder, scene_name))

            for image_name in image_list:
                if "." in image_name:
                    continue

                self.image_path_list.append(os.path.join(img_folder, scene_name, image_name, "rgb.png"))

    def __getitem__(self, idx):

        img_path = self.image_path_list[idx]
        temperature_path = self.image_path_list[idx].replace('rgb.png', 'temperature.npy')
        mask_path = self.image_path_list[idx].replace('rgb.png', 'mask.png')
        boundary_path = self.image_path_list[idx].replace('rgb.png', 'boundary.png')

        img = Image.open(img_path).convert('RGB')

        temperature = np.load(temperature_path)
        temperature = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature)) * 255

        temperature = Image.fromarray(temperature).convert('RGB')

        w, h = img.size

        masks = Image.open(mask_path).convert('L')

        # cv2.imshow("old", np.asarray(img))

        # if "train" in img_path:
        #     if random.uniform(0, 1) < 0.5:
        #         img = merge(img, masks, self.patch_path_list)

        # cv2.imshow("new", np.asarray(img))
        # cv2.waitKey(0)

        masks = np.asarray(masks, dtype=np.uint32) / 255
        masks = torch.as_tensor(masks, dtype=torch.uint8).unsqueeze(0)

        boundaries = np.asarray(Image.open(boundary_path).convert('L'), dtype=np.uint32) / 255
        boundaries = torch.as_tensor(boundaries, dtype=torch.uint8).unsqueeze(0)

        target = {}
        target['image_id'] = torch.tensor(idx)

        target['masks'] = masks
        target['orig_masks'] = masks

        target['edges'] = boundaries
        target['orig_edges'] = boundaries

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])

        if self.transforms is not None:
            img, temperature, target = self.transforms(img, temperature, target)

        return img, temperature, target, img_path

    def __len__(self):
        return len(self.image_path_list)

    # def get_height_and_width(self, idx):
    #     img_info = self.coco['images'][idx]
    #     height = img_info['height']
    #     width = img_info['width']
    #     return height, width


def build(image_set, args):
    img_folder_root = args.rgbt_path

    PATHS = {
        "train": "train",
        "val": "test",
    }

    img_folder = PATHS[image_set]
    img_folder_path = os.path.join(img_folder_root, img_folder)

    dataset = RGBT(img_folder_path, transforms=make_rgbt_transforms(image_set),
                   return_masks=args.masks)

    return dataset


def make_rgbt_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [224, 256, 288, 320, 352, 384, 416, 448, 480]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    # T.RandomResize([300, 400, 500]),
                    T.RandomSizeCrop(320, 420),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([480], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    if max_x == 0:
        x = 0
    else:
        x = np.random.randint(0, max_x)

    if max_y == 0:
        y = 0
    else:
        y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def merge(rgb, mask, patch_path_list):
    rgb = np.asarray(rgb)
    mask = np.asarray(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ori_rgb = rgb

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        start_x = x
        start_y = y

        end_x = x + w
        end_y = y + h

        patch_rgb = ori_rgb[start_y:end_y, start_x:end_x, :]

        single_mask = np.zeros_like(rgb)
        approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
        cv2.fillPoly(single_mask, [approx], (255, 255, 255))

        patch_mask = single_mask[start_y:end_y, start_x:end_x, :]

        # original_patch_mask = np.expand_dims(mask[start_y:end_y, start_x:end_x], -1).repeat(3, -1)
        #
        # boolean_patch_mask = patch_mask[:, :, 0] > 127
        # boolean_original_patch_mask = original_patch_mask[:, :, 0] > 127

        # cv2.imshow("patch_mask", patch_mask)
        # cv2.imshow("original_patch_mask", original_patch_mask)
        # cv2.waitKey(0)

        # to avoid replacing the inner contour of a glass region, which is usually the doorknob
        # if np.sum(np.logical_and(boolean_patch_mask, boolean_original_patch_mask)) / np.sum(
        #         np.logical_or(boolean_patch_mask, boolean_original_patch_mask)) < 0.6:
        #     rgb[start_y:end_y, start_x:end_x, :] = np.where(original_patch_mask == 255, patch_rgb, rgb[start_y:end_y, start_x:end_x, :])
        #
        #     continue

        patch_glass = cv2.imread(random.choice(patch_path_list))

        gh, gw, _ = patch_glass.shape
        mh, mw, _ = patch_mask.shape

        if mh < gh and mw < gw:
            pass
        elif mh < gh and mw >= gw:
            patch_glass = cv2.resize(patch_glass, (mw, int(gh / gw * mw) + 1), cv2.INTER_LINEAR)
        elif mh >= gh and mw < gw:
            patch_glass = cv2.resize(patch_glass, (int(gw / gh * mh) + 1, mh), cv2.INTER_LINEAR)
        else:
            if int(gw / gh * mh) + 1 >= mw:
                patch_glass = cv2.resize(patch_glass, (int(gw / gh * mh) + 1, mh), cv2.INTER_LINEAR)
            else:
                patch_glass = cv2.resize(patch_glass, (mw, int(gh / gw * mw) + 1), cv2.INTER_LINEAR)

        patch_glass = get_random_crop(patch_glass, mh, mw)

        # patch_rgb = Image.fromarray(patch_rgb).convert('RGBA')
        # patch_glass = Image.fromarray(patch_glass).convert('RGBA')
        # patch_rgb = Image.composite(patch_rgb, patch_glass, patch_mask[:, :, 0] > 127)
        # patch_rgb = np.asarray(patch_rgb)

        blurred_mask = cv2.GaussianBlur(patch_mask, ksize=(13, 13), sigmaX=2) / 255

        # cv2.imshow("blurred_mask", blurred_mask)
        # cv2.waitKey(0)

        # patch_rgb = np.where(patch_mask == 255, patch_glass, patch_rgb)

        patch_rgb = blurred_mask * patch_glass + (1 - blurred_mask) * patch_rgb

        if random.uniform(0, 1) < 0.5:
            rgb[start_y:end_y, start_x:end_x, :] = patch_rgb

    return Image.fromarray(rgb).convert('RGB')
