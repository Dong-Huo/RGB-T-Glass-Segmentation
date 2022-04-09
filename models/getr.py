# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional
from torch import Tensor
import numpy as np
import pytorch_iou
import pytorch_ssim
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .double_transformer import build_transformer


class GETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, rgb_backbone, temperature_backbone, transformer):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        # self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.rgb_input_proj = nn.Conv2d(rgb_backbone.num_channels, hidden_dim, kernel_size=1)
        self.temperature_input_proj = nn.Conv2d(temperature_backbone.num_channels, hidden_dim, kernel_size=1)
        self.rgb_backbone = rgb_backbone
        self.temperature_backbone = temperature_backbone
        self.mask_head = MaskHeadSmallConv(hidden_dim * 2, [2048, 1024, 512, 256], hidden_dim)
        # self.fusion_embedding = nn.Embedding(, hidden_dim)

    def forward(self, rgb: NestedTensor, temperature: NestedTensor, final_size, gate_flag: bool):

        if isinstance(rgb, (list, torch.Tensor)):
            rgb = nested_tensor_from_tensor_list(rgb)

        if isinstance(temperature, (list, torch.Tensor)):
            temperature = nested_tensor_from_tensor_list(temperature)

        rgb_features, rgb_pos = self.rgb_backbone(rgb)
        temperature_features, temperature_pos = self.temperature_backbone(temperature)

        rgb_src, rgb_mask = rgb_features[-1].decompose()
        temperature_src, temperature_mask = temperature_features[-1].decompose()

        assert rgb_mask is not None
        assert temperature_mask is not None

        rgb_proj = self.rgb_input_proj(rgb_src)
        temperature_proj = self.temperature_input_proj(temperature_src)

        fusion_memory = self.transformer(rgb_proj, rgb_mask, rgb_pos[-1], temperature_proj, temperature_mask,
                                         temperature_pos[-1], gate_flag)

        out = {}
        mask_list, edge_list = self.mask_head(fusion_memory,
                                              [rgb_features[3].tensors, rgb_features[2].tensors,
                                               rgb_features[1].tensors, rgb_features[0].tensors],
                                              [temperature_features[3].tensors, temperature_features[2].tensors,
                                               temperature_features[1].tensors, temperature_features[0].tensors],
                                              final_size, gate_flag)

        out['pred_mask1'] = mask_list[0]
        out['pred_mask2'] = mask_list[1]
        out['pred_mask3'] = mask_list[2]
        out['pred_mask4'] = mask_list[3]
        out['pred_mask5'] = mask_list[4]
        out['pred_mask6'] = mask_list[5]

        out['pred_edge1'] = edge_list[0]
        out['pred_edge2'] = edge_list[1]
        out['pred_edge3'] = edge_list[2]
        out['pred_edge4'] = edge_list[3]
        out['pred_edge5'] = edge_list[4]
        out['pred_edge6'] = edge_list[5]

        return out


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]

        # self.conv_block1 = nn.Sequential(nn.Conv2d(dim, inter_dims[1], 3, padding=1),
        #                                  nn.BatchNorm2d(inter_dims[1]),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(inter_dims[1], inter_dims[1], 3, padding=1),
        #                                  nn.BatchNorm2d(inter_dims[1]),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(inter_dims[1], inter_dims[1], 3, padding=1),
        #                                  nn.BatchNorm2d(inter_dims[1]),
        #                                  nn.ReLU(inplace=True))
        #
        # self.conv_block2 = nn.Sequential(nn.Conv2d(dim + inter_dims[1], inter_dims[1], 3, padding=1),
        #                                  nn.BatchNorm2d(inter_dims[1]),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(inter_dims[1], inter_dims[1], 3, padding=1),
        #                                  nn.BatchNorm2d(inter_dims[1]),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(inter_dims[1], inter_dims[1], 3, padding=1),
        #                                  nn.BatchNorm2d(inter_dims[1]),
        #                                  nn.ReLU(inplace=True))

        self.inference_module1 = InferenceModule(inter_dims[1], inter_dims[1])
        self.inference_module2 = InferenceModule(inter_dims[1], inter_dims[2])
        self.inference_module3 = InferenceModule(inter_dims[2], inter_dims[3])
        self.inference_module4 = InferenceModule(inter_dims[3], inter_dims[4])
        self.inference_module5 = InferenceModule(inter_dims[4], inter_dims[5])

        self.rgb_adapter1 = nn.Sequential(nn.Conv2d(fpn_dims[0], inter_dims[1], 1),
                                          nn.BatchNorm2d(inter_dims[1]),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter2 = nn.Sequential(nn.Conv2d(fpn_dims[1], inter_dims[1], 1),
                                          nn.BatchNorm2d(inter_dims[1]),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter3 = nn.Sequential(nn.Conv2d(fpn_dims[2], inter_dims[2], 1),
                                          nn.BatchNorm2d(inter_dims[2]),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter4 = nn.Sequential(nn.Conv2d(fpn_dims[3], inter_dims[3], 1),
                                          nn.BatchNorm2d(inter_dims[3]),
                                          nn.ReLU(inplace=True))

        self.temperature_adapter1 = nn.Sequential(nn.Conv2d(fpn_dims[0], inter_dims[1], 1),
                                                  nn.BatchNorm2d(inter_dims[1]),
                                                  nn.ReLU(inplace=True))
        self.temperature_adapter2 = nn.Sequential(nn.Conv2d(fpn_dims[1], inter_dims[1], 1),
                                                  nn.BatchNorm2d(inter_dims[1]),
                                                  nn.ReLU(inplace=True))
        self.temperature_adapter3 = nn.Sequential(nn.Conv2d(fpn_dims[2], inter_dims[2], 1),
                                                  nn.BatchNorm2d(inter_dims[2]),
                                                  nn.ReLU(inplace=True))
        self.temperature_adapter4 = nn.Sequential(nn.Conv2d(fpn_dims[3], inter_dims[3], 1),
                                                  nn.BatchNorm2d(inter_dims[3]),
                                                  nn.ReLU(inplace=True))

        self.pa_module1 = PixelAttention(inter_dims[1], 3)
        self.pa_module2 = PixelAttention(inter_dims[1], 3)
        self.pa_module3 = PixelAttention(inter_dims[2], 3)
        self.pa_module4 = PixelAttention(inter_dims[3], 3)

        self.mask_out_conv = nn.Conv2d(5, 1, 1)
        self.edge_out_conv = nn.Conv2d(5, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fusion_memory, rgb_fpns: List[Tensor], temperature_fpns: List[Tensor], final_size, gate_flag):

        x = fusion_memory

        rgb_cur_fpn = self.rgb_adapter1(rgb_fpns[0])
        temperature_cur_fpn = self.temperature_adapter1(temperature_fpns[0])
        edge1, mask1, x = self.inference_module1(self.pa_module1(x, rgb_cur_fpn, temperature_cur_fpn, gate_flag))  # 1/32
        edge1 = F.interpolate(edge1, size=final_size, mode="bilinear")
        mask1 = F.interpolate(mask1, size=final_size, mode="bilinear")

        rgb_cur_fpn = self.rgb_adapter2(rgb_fpns[1])
        temperature_cur_fpn = self.temperature_adapter2(temperature_fpns[1])
        x = F.interpolate(x, size=rgb_cur_fpn.shape[-2:], mode="bilinear")
        edge2, mask2, x = self.inference_module2(self.pa_module2(x, rgb_cur_fpn, temperature_cur_fpn, gate_flag))  # 1/16
        edge2 = F.interpolate(edge2, size=final_size, mode="bilinear")
        mask2 = F.interpolate(mask2, size=final_size, mode="bilinear")

        rgb_cur_fpn = self.rgb_adapter3(rgb_fpns[2])
        temperature_cur_fpn = self.temperature_adapter3(temperature_fpns[2])
        x = F.interpolate(x, size=rgb_cur_fpn.shape[-2:], mode="bilinear")
        edge3, mask3, x = self.inference_module3(self.pa_module3(x, rgb_cur_fpn, temperature_cur_fpn, gate_flag))  # 1/8
        edge3 = F.interpolate(edge3, size=final_size, mode="bilinear")
        mask3 = F.interpolate(mask3, size=final_size, mode="bilinear")

        rgb_cur_fpn = self.rgb_adapter4(rgb_fpns[3])
        temperature_cur_fpn = self.temperature_adapter4(temperature_fpns[3])
        x = F.interpolate(x, size=rgb_cur_fpn.shape[-2:], mode="bilinear")
        edge4, mask4, x = self.inference_module4(self.pa_module4(x, rgb_cur_fpn, temperature_cur_fpn, gate_flag))  # 1/4
        edge4 = F.interpolate(edge4, size=final_size, mode="bilinear")
        mask4 = F.interpolate(mask4, size=final_size, mode="bilinear")

        x = F.interpolate(x, size=final_size, mode="bilinear")
        edge5, mask5, _ = self.inference_module5(x)  # 1/1

        mask_list = [mask1, mask2, mask3, mask4, mask5]
        edge_list = [edge1, edge2, edge3, edge4, edge5]

        final_mask = self.mask_out_conv(torch.cat(mask_list, 1))
        final_edge = self.mask_out_conv(torch.cat(edge_list, 1))

        mask_list.append(final_mask)
        edge_list.append(final_edge)

        return [torch.sigmoid(x) for x in mask_list], [torch.sigmoid(x) for x in edge_list]


class PixelAttention(nn.Module):
    def __init__(self, inchannels, times):
        super(PixelAttention, self).__init__()

        self.mask_conv1 = nn.Sequential(nn.Conv2d(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(inchannels),
                                       nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(inchannels),
                                       nn.Conv2d(inchannels, 1, 1))

        self.mask_conv2 = nn.Sequential(nn.Conv2d(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rgb, temperature, gate_flag):

        mask1 = self.mask_conv1(torch.cat([x, rgb], 1))
        mask1 = torch.sigmoid(mask1)
        rx = x + rgb * mask1

        mask2 = self.mask_conv2(torch.cat([x, temperature], 1))
        mask2 = torch.sigmoid(mask2)

        if gate_flag:
            x = rx + temperature * mask2
        else:
            x = rx

        return x


class ConvMpnModel(nn.Module):
    def __init__(self, inchannels, out_channels):
        super(ConvMpnModel, self).__init__()
        assert inchannels >= out_channels
        self.out_channels = out_channels
        self.seq = nn.Sequential(
            # nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(inchannels),
            # nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(inchannels),
            # nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(inchannels),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inchannels),
            nn.Conv2d(inchannels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.seq(x)


class InferenceModule(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, indim, outdim):
        super().__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(indim, outdim, 3, padding=1),
                                        nn.BatchNorm2d(outdim),
                                        nn.ReLU(inplace=True))

        self.edge_conv = nn.Sequential(nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True))

        self.mask_conv = nn.Sequential(nn.Conv2d(outdim * 2, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True))

        self.out_mask_lay = torch.nn.Conv2d(outdim, 1, 3, padding=1)
        self.out_edge_lay = torch.nn.Conv2d(outdim, 1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_block(x)
        edge_feature = self.edge_conv(x)
        edge = self.out_edge_lay(edge_feature)
        x = self.mask_conv(torch.cat([edge_feature, x], 1))
        mask = self.out_mask_lay(x)

        return edge, mask, x


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ssim = pytorch_ssim.SSIM(window_size=11, size_average=True)
        self.iou_loss = pytorch_iou.IOU(size_average=True)

    def forward(self, outputs, targets):
        masks = [t["masks"] for t in targets]
        edges = [t["edges"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, mask_valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(outputs["pred_mask1"])

        target_edges, edge_valid = nested_tensor_from_tensor_list(edges).decompose()
        target_edges = target_edges.to(outputs["pred_edge1"])

        # upsample predictions to the target size

        losses = {}

        for i in range(1, len(outputs) // 2 + 1):
            src_mask = outputs["pred_mask" + str(i)]
            # target_m = F.interpolate(target_masks, src_mask.shape[2:], mode='nearest')
            # mask_v = F.interpolate(mask_valid.unsqueeze(1).repeat(1, src_mask.shape[1], 1, 1).to(torch.uint8),
            #                        src_mask.shape[2:], mode='nearest').to(torch.bool)

            src_edge = outputs["pred_edge" + str(i)]
            # target_e = F.interpolate(target_edges, src_edge.shape[2:], mode='nearest')
            # edge_v = F.interpolate(edge_valid.unsqueeze(1).repeat(1, src_edge.shape[1], 1, 1).to(torch.uint8),
            #                        src_edge.shape[2:], mode='nearest').to(torch.bool)

            # src_mask = torch.where(mask_v == False, src_mask, torch.zeros_like(src_mask))
            # src_edge = torch.where(edge_v == False, src_edge, torch.zeros_like(src_edge))

            bce_loss = self.bce_loss(src_mask, target_masks) * self.weight_dict['loss_bce']
            # iou_loss = self.iou_loss(src_mask, target_m) * self.weight_dict['loss_iou']
            # ssim_loss = (1 - self.ssim(src_mask, target_m)) * self.weight_dict['loss_ssim']
            edge_loss = self.bce_loss(src_edge, target_edges) * self.weight_dict['loss_edge']

            losses['bce_loss' + str(i)] = bce_loss
            # losses['iou_loss' + str(i)] = iou_loss
            # losses['ssim_loss' + str(i)] = ssim_loss
            # losses['edge_loss' + str(i)] = edge_loss

        return losses


def build(args):
    device = torch.device(args.device)

    rgb_backbone = build_backbone(args)
    temperature_backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = GETR(
        rgb_backbone,
        temperature_backbone,
        transformer,
    )

    # weight_dict = {'loss_bce': 1, 'loss_ssim': 1, 'loss_iou': 1, 'loss_edge': 1}

    weight_dict = {'loss_bce': 1, 'loss_edge': 1}

    criterion = SetCriterion(weight_dict)
    criterion.to(device)

    postprocessors = {}

    postprocessors['mask'] = PostProcessMask()

    return model, criterion, postprocessors


class PostProcessMask(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_mask6"]
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        # outputs_masks = (outputs_masks > self.threshold).cpu()

        results = []

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]

            cropped_mask = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            # cropped_mask = F.interpolate(cropped_mask.float(), size=tuple(tt.tolist()), mode="nearest").byte()
            cropped_mask = F.interpolate(cropped_mask.float(), size=tuple(tt.tolist()), mode="nearest")
            results.append(cropped_mask)

        return results
