# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision


def build_dataset(image_set, args):

    if args.dataset_file == 'rgbt':
        from .rgbt import build as build_rgbt
        return build_rgbt(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
