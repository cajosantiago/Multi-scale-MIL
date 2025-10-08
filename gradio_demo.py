import numpy as np
import pandas as pd
from pathlib import Path
import os

import torch
import torchvision
import torch.nn.functional as F

from MIL import build_model
# from utils.generic_utils import seed_all
import argparse
import yaml
from pathlib import Path
import gdown
import tarfile
import requests
from PIL import Image
import gradio as gr

os.environ['GRADIO_SERVER_NAME'] = "0.0.0.0"

def config():
    parser = argparse.ArgumentParser()

    # Folders
    parser.add_argument('--output_dir', metavar='DIR', default='Mammo-CLIP-output/out_splits_new',
                        help='path to output logs')
    parser.add_argument("--data_dir", default="datasets/Vindir-mammoclip", type=str, help="Path to data file")
    parser.add_argument("--clip_chk_pt_path", default='checkpoints/b2-model-best-epoch-10.tar', type=str, help="Path to Mammo-CLIP chkpt")
    parser.add_argument("--csv_file", default="grouped_df.csv", type=str, help="data csv file")
    parser.add_argument('--feat_dir', default='new_extracted_features', type=str)
    parser.add_argument("--img_dir", default="test_image.png", type=str,
                        help="Path to image file")

    parser.add_argument('--train', action='store_true', default=False, help='Training mode.')
    parser.add_argument('--evaluation', action='store_true', default=False, help='Evaluation mode.')
    parser.add_argument('--eval_set', default='test', choices=['val', 'test'], type=str, help="")

    # Data settings
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--dataset", default="ViNDr", type=str, help="Dataset name.")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument("--label", default="Suspicious_Calcification", type=str, help="Mass or Suspicious_Calcification")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--start_run", default=0, type=int)
    parser.add_argument('--val_split', type=float, default=0.2, help='val split ratio (default: 0.2)')
    parser.add_argument("--n_folds", default=1, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)

    # Mammo-CLIP settings
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str,
                        help="For b5 classification, [upmc_breast_clip_det_b5_period_n_lp for linear probe and  upmc_breast_clip_det_b5_period_n_ft for finetuning]. "
                             "For b2 classification, [upmc_breast_clip_det_b2_period_n_lp for linear probe and  upmc_breast_clip_det_b2_period_n_ft for finetuning].")
    parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str)
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    parser.add_argument("--swin_model_type", default="y", type=str)

    parser.add_argument("--feature_extraction", default='online', type=str)
    parser.add_argument("--feat_dim", default=352, type=int)

    # Patch extraction
    parser.add_argument('--patching', action='store_true', default=False,
                        help='Wether to perform patching on full-resolution images. If false, it will consider previously extracted patches that were saved in a directory (default: False)')
    parser.add_argument('--source_image', type=str, default='patches', choices=['patches', 'full_image'])
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--overlap', type=float, nargs='*', default=[0.75])

    # MIL model parameters
    parser.add_argument('--mil_type', default='pyramidal_mil', choices=[None, 'instance', 'embedding', 'pyramidal_mil'], type=str,
                        help="MIL approach")
    parser.add_argument('--pooling_type', default='pma',
                        choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str,
                        help="MIL pooling operator")
    parser.add_argument('--type_mil_encoder', default='isab', choices=['mlp', 'sab', 'isab'], type=str,
                        help="Type of MIL encoder.")

    parser.add_argument('--fcl_attention_dim', type=int, default=128, metavar='N',
                        help='parameter for attention (internal hidden units)')
    parser.add_argument('--map_prob_func', type=str, default='softmax',
                        choices=['softmax', 'sparsemax', 'entmax', 'alpha_entmax'])

    parser.add_argument('--fcl_encoder_dim', type=int, default=256,
                        help='parameter for set transformer (internal hidden units)')
    parser.add_argument('--sab_num_heads', type=int, default=4,
                        help='parameter for set transformer (number of self-attention heads in set attention blocks)')
    parser.add_argument('--isab_num_heads', type=int, default=4,
                        help='parameter for set transformer (number of self-attention heads in induced set attention blocks)')
    parser.add_argument('--pma_num_heads', type=int, default=1,
                        help='parameter for set transformer (number of self-attention heads in pooling by multihead attention)')
    parser.add_argument('--num_encoder_blocks', type=int, default=2,
                        help='parameter for set transformer (number of encoder layers)')
    parser.add_argument('--trans_num_inds', type=int, default=20,
                        help='parameter for set transformer (number of inducing points for the ISAB)')
    parser.add_argument('--trans_layer_norm', type=bool, default=True)

    # Multi-scale MIL
    parser.add_argument('--multi_scale_model', type=str, choices=['fpn', 'backbone_pyramid', 'msp'], default='fpn')
    parser.add_argument('--scales', type=int, nargs='*', default=(16, 32, 128),
                        help="List of scales to use for the multi-scale model.")

    parser.add_argument('--fpn_dim', type=int, default=256)
    parser.add_argument('--upsample_method', type=str, choices=['bilinear', 'nearest'], default='nearest')
    parser.add_argument('--norm_fpn', type=bool, default=False)

    parser.add_argument('--deep_supervision', action='store_true', default=False)
    parser.add_argument('--type_scale_aggregator', type=str,
                        choices=['concatenation', 'max_p', 'mean_p', 'attention', 'gated-attention'], default='gated-attention')

    # Nested MIL
    parser.add_argument('--nested_model', action='store_true', default=False)
    parser.add_argument('--type_region_aggregator', type=str,
                        choices=['concatenation', 'max_p', 'mean_p', 'attention', 'gated-attention'], default=None)
    parser.add_argument('--type_region_encoder', default=None, choices=['mlp', 'sab', 'isab'], type=str,
                        help="Type of MIL encoder.")
    parser.add_argument('--type_region_pooling', default=None,
                        choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str,
                        help="MIL pooling operator")
    parser.add_argument('--model_dir', default='./models/',)

    # Data augmentation settings
    parser.add_argument("--balanced-dataloader", default='n', type=str,
                        help='Enable weighted sampling during training (default: False).')
    parser.add_argument("--data_aug", action='store_true', default=False)

    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)

    # LR scheduler settings
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)

    # Regularization parameters
    parser.add_argument('--drop_classhead', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate used in the classification head (default: 0.)')
    parser.add_argument('--drop_attention_pool', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    parser.add_argument('--drop_mha', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    parser.add_argument('--fcl_dropout', type=float, default=0.0)
    parser.add_argument("--lamda", type=float, default=0.0,
                        help='lambda used for balancing cross-entropy loss and rank loss.')

    # ROI evaluation parameters
    parser.add_argument('--roi_eval', action='store_true', default=False,
                        help='Evaluate post-hoc detection performance')
    parser.add_argument('--roi_attention_threshold', type=float, default=0.5)
    parser.add_argument('--visualize_num_images', default=0, type=int, help="")
    parser.add_argument('--quantile_threshold', default=0.95, type=float)
    parser.add_argument('--max_bboxes', default=100, type=int)
    parser.add_argument('--min_area', default=1024, type=int)
    parser.add_argument('--iou_threshold', default=0.25, type=float)

    parser.add_argument('--roi_eval_scheme', default='all_roi',
                        choices=['small_roi', 'medium_roi', 'large_roi', 'all_roi'], type=str, help="")
    parser.add_argument('--roi_eval_set', default='test', choices=['val', 'test'], type=str, help="")

    parser.add_argument('--iou_method', default='iou', choices=['iou', 'iobb_detection', 'iobb_annotation'], type=str)
    parser.add_argument('--ap_method', default='area', choices=['area', '11points'], type=str)

    # Device settings
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)

    # Misc
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument('--eval_scheme', default='kruns_train+val', type=str,
                        help='Evaluation scheme [kruns_train+val | kfold_cv+test ]')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--test_example', default=None, type=str)

    return parser.parse_args()


class Patching:
    """
    Extracts patches from an image.

    Args:
        patch_size (int): Default patch size (used if multi_scale_model is None).
        overlap (float or list): Overlap between patches.
        multi_scale_model (str or None): One of ['msp', 'fpn', 'backbone_pyramid'], or None for single scale.
        scales (list): List of scales to use if multi_scale_model == 'msp'.
        mean (float): Mean pixel value used for normalization (for padding).
        std (float): Standard deviation used for normalization (for padding).
    """

    def __init__(self, patch_size=512, overlap=0, multi_scale_model=None, scales=[16, 8, 4], mean=0.3089279,
                 std=0.25053555408335154):
        self.patch_size = patch_size if multi_scale_model is not None else scales[0]
        self.overlap = overlap
        self.multi_scale_model = multi_scale_model
        self.scales = scales

        self.mean = mean
        self.std = std

    def extract_patch(self, image, x_start, y_start, size, img_h, img_w):
        """
        Extracts a single patch from the image and pads it if it extends beyond image boundaries.
        """

        # Define patch bounds
        x_end = x_start + size
        y_end = y_start + size

        # Compute the effective patch size
        x_pad_start = max(0, -x_start)
        y_pad_start = max(0, -y_start)
        x_pad_end = max(0, x_end - img_w)
        y_pad_end = max(0, y_end - img_h)

        # Ensure the starting and ending positions are within the image boundaries
        x_start_clipped = max(0, x_start)
        y_start_clipped = max(0, y_start)
        x_end_clipped = min(img_w, x_end)
        y_end_clipped = min(img_h, y_end)

        # Extract the valid region of the patch from the image
        patch = image[:, y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

        # Normalize padding value (black pixel normalized)
        normalized_black_value = (0 - self.mean) / self.std

        # Pad to match patch size if needed
        patch = F.pad(
            patch,
            pad=(x_pad_start, x_pad_end, y_pad_start, y_pad_end),
            mode='constant',
            value=normalized_black_value
        )

        return patch, x_start, y_start

    def __call__(self, img_array_with_padding):
        # Unpack the input
        img_array, padding = img_array_with_padding
        c, img_height, img_width = img_array.shape

        # Initialize dictionaries for patches and coordinates
        if self.multi_scale_model == 'msp':
            patches = {size: [] for size in self.scales}
            patch_coords = {size: [] for size in self.scales}
        else:
            patches = []
            patch_coords = []

        if self.multi_scale_model == 'msp':
            # Multi-scale patching
            for idx, patch_size in enumerate(self.scales):
                step_size = patch_size - int(patch_size * self.overlap[idx])
                # step_size = 64

                start_x, start_y, w, h = (0, 0, img_width, img_height)
                x_range = range(0, img_width, step_size)
                y_range = range(0, img_height, step_size)

                for x in x_range:
                    for y in y_range:
                        patch, x_start, y_start = self.extract_patch(img_array, x, y, patch_size, h, w)

                        patches[patch_size].append(patch)
                        patch_coords[patch_size].append([int(x_start), int(y_start)])

        elif self.multi_scale_model in ['fpn', 'backbone_pyramid']:
            step_size = self.patch_size - int(self.patch_size * self.overlap[0])

            start_x, start_y, w, h = (0, 0, img_width, img_height)
            stop_y = min(start_y + h, h - self.patch_size + 1)
            stop_x = min(start_x + w, w - self.patch_size + 1)
            x_range = np.arange(start_x, stop_x, step=step_size)
            y_range = np.arange(start_y, stop_y, step=step_size)

            for x in x_range:
                for y in y_range:
                    patch = img_array[:, y:y + self.patch_size, x:x + self.patch_size]

                    patches.append(patch)
                    patch_coords.append([int(x), int(y)])

        else:
            # Standard single-scale patching
            patch_size = self.scales[0]
            step_size = patch_size - int(patch_size * self.overlap[0])

            start_x, start_y, w, h = (0, 0, img_width, img_height)
            x_range = range(0, img_width, step_size)
            y_range = range(0, img_height, step_size)

            for x in x_range:
                for y in y_range:
                    patch, x_start, y_start = self.extract_patch(img_array, x, y, patch_size, h, w)

                    patches.append(patch)
                    patch_coords.append([int(x_start), int(y_start)])

        # Convert lists to tensors or arrays
        if self.multi_scale_model == 'msp':
            patch_coords = {size: np.array(patch_coords[size]) for size in patch_coords}
            patches = {size: torch.stack(patches[size]) for size in patches}

        else:
            patch_coords = np.array(patch_coords)
            patches = torch.stack(patches)

        # Sort patch coordinates and patches
        if isinstance(patch_coords, dict):  # Multi-scale image pyramid
            for size in patch_coords:
                sorted_indices = np.lexsort((patch_coords[size][:, 0], patch_coords[size][:, 1]))  # Sort by y, then x
                patch_coords[size] = patch_coords[size][sorted_indices]
                patches[size] = patches[size][sorted_indices]
        else:
            sorted_indices = np.lexsort((patch_coords[:, 0], patch_coords[:, 1]))  # Sort by y, then x
            patch_coords = patch_coords[sorted_indices]
            patches = patches[sorted_indices]

        return patches, patch_coords, padding


def pad_image(img_array, patch_size, mean, std):
    """
    Pads an image tensor so that its height and width are multiples of the patch size.

    Args:
        img_array (Tensor): Input image tensor with shape (C, H, W).
        patch_size (int): patch size.
        mean (float): Mean used for normalization.
        std (float): Standard deviation used for normalization.

    Returns:
        padded_img (Tensor): Padded image tensor.
        padding (tuple): Tuple of applied paddings: (left, right, top, bottom).
    """

    # Get dimensions of the image
    if len(img_array.size()) == 3:  # If image has channel dimension
        c, h, w = img_array.size()
    else:  # Just height and width
        h, w = img_array.size()

    # Compute new dimensions that are divisible by patch_size
    new_h = h + (patch_size - h % patch_size)
    new_w = w + (patch_size - w % patch_size)

    # Determine needed padding for width and height
    additional_h = new_h - h
    additional_w = new_w - w

    # Initialize padding amounts
    padding_left, padding_right, padding_top, padding_bottom = 0, 0, 0, 0

    # Horizontal sum (sums over height)
    horizontal_sum = img_array.sum(axis=(0, 1))
    left_info = horizontal_sum[:w // 2].sum()
    right_info = horizontal_sum[w // 2:].sum()

    # Apply padding on the side with less information for width
    if left_info < right_info:
        padding_left = additional_w
    else:
        padding_right = additional_w

    # Vertical sum (sums over width)
    vertical_sum = img_array.sum(axis=(0, 2))
    top_info = vertical_sum[:h // 2].sum()
    bottom_info = vertical_sum[h // 2:].sum()

    # Apply padding on the side with less information for height
    if top_info < bottom_info:
        padding_top = additional_h
    else:
        padding_bottom = additional_h

    # Construct the padding configuration
    normalized_black_value = (0.0 - mean) / std  # Compute padding value (normalized black)
    padded_img = F.pad(img_array,
                       (padding_left, padding_right, padding_top, padding_bottom),
                       mode='constant',
                       value=normalized_black_value
                       )
    return padded_img, (padding_left, padding_right, padding_top, padding_bottom)

class lambda_funct(torchvision.transforms.Lambda):
    """
    Lambda transform wrapper for padding an image.

    Args:
        lambd (callable): Padding function to apply.
        patch_size (int): Target patch size to pad to.
        mean (float): Mean for normalization.
        std (float): Std for normalization.
    """

    def __init__(self, lambd, patch_size, mean, std):
        super().__init__(lambda_funct)

        self.lambd = lambd
        self.patch_size = patch_size
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return self.lambd(img, self.patch_size, self.mean, self.std)

def extract_bounding_boxes_from_heatmap(heatmap, quantile_threshold=0.98, max_bboxes=3, min_area=230,
                                        iou_threshold=0.5):
    """
    Extract bounding boxes from a heatmap by thresholding high-attention regions on a given heatmap.

    Args:
        heatmap (np.ndarray): 2D heatmap.
        quantile_threshold (float): Quantile threshold for heatmap binarization (default: 0.98).
        max_bboxes (int): Maximum number of bounding boxes to return after NMS (default: 3).
        min_area (int): Minimum area (in pixels) for a connected component to be considered (default: 230).
        iou_threshold (float, optional): IoU threshold used during NMS and overlap filtering (default: 0.5).

    Returns:
        list: List of bounding boxes with scores, in the format [x_min, y_min, x_max, y_max, score].
    """

    # Threshold heatmap based on quantile and minimum value
    q = np.quantile(heatmap, quantile_threshold)
    mask = (heatmap > q) & (heatmap > 0.5)

    # label connected pixels in the binary mask
    label_im, nb_labels = ndimage.label(mask)

    # find the sizes of connected pixels
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # Remove connected components smaller than min_area
    mask_size = sizes < min_area
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    # Re-label after removing small components
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)  # sort objects from large to small

    # generate bounding boxes
    bboxes = []
    for l in range(1, len(labels)):
        slice_x, slice_y = ndimage.find_objects(label_im == l)[0]

        # Validate bounding box dimensions
        if (slice_x.start < slice_x.stop) & (slice_y.start < slice_y.stop):

            if (slice_x.stop - slice_x.start) * (slice_y.stop - slice_y.start) < min_area:
                continue

            b = [slice_y.start, slice_x.start, slice_y.stop, slice_x.stop]
            score = get_cumlative_attention(heatmap, b)

            bboxes.append([slice_y.start, slice_x.start, slice_y.stop, slice_x.stop, score])

    # Sort boxes by score descending
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    # Convert to tensor for NMS if there are any detections
    if len(bboxes) > 0:
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        # Apply Non-Maximum Suppression to reduce overlapping boxes
        keep_indices = nms(bboxes_tensor[:, :4], bboxes_tensor[:, 4], iou_threshold)
        keep_indices = keep_indices[:max_bboxes]
        bboxes = bboxes_tensor[keep_indices]

        # remove boxes contain within others
        to_keep = []
        for i in range(len(bboxes)):
            keep = True
            for j in range(len(bboxes)):
                if i != j:
                    box1 = bboxes[i, :4]
                    box2 = bboxes[j, :4]

                    # Compute intersection
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])

                    intersection = max(0, x2 - x1) * max(0, y2 - y1)

                    # Compute areas
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                    # Check if intersection equals the area of the smaller box
                    if intersection >= area1 * iou_threshold:
                        keep = False
                        break
            if keep:
                to_keep.append(bboxes[i])

        # Convert back to list
        bboxes = torch.stack(to_keep).tolist() if to_keep else []

    return bboxes

def visualize_detection(args, model, img, bag_coords, bag_info):
    img_h, img_w = img.shape[:2]

    # Get instance-level attention scores for all scales from the model
    scale_attentions_dict = model.get_patch_scores()

    # If scale aggregator uses concatenation or gated-attention, also get scale scores
    if args.type_scale_aggregator in ['concatenation', 'gated-attention']:
        scale_scores = model.get_scale_scores().detach().cpu()

    # Initialize multi-scale aggregated heatmap
    aggregated_heatmap = torch.zeros(img_h, img_w)

    # Dictionary to store heatmaps and bounding boxes per scale
    heatmaps = {}

    # Loop over each scale and its corresponding attention scores
    for idx, (scale, attention_scores) in enumerate(scale_attentions_dict.items()):

        # Get scale-level weight depending on aggregator type
        if args.type_scale_aggregator == 'gated-attention':
            scale_score = scale_scores[0, idx]

        elif args.type_scale_aggregator == 'concatenation':
            scale_score = scale_scores.squeeze()[idx]

        attention_scores = attention_scores.detach().cpu().squeeze()

        # Handle coordinate and patch size depending on multi-scale model type
        if args.multi_scale_model == 'msp':
            bag_coords_scale = bag_coords[scale] if scale != 'aggregated' else bag_coords[args.scales[0]]
            patch_size = bag_info[scale]['patch_size'] if scale != 'aggregated' else args.scales[0]

        elif args.multi_scale_model in ['fpn', 'backbone_pyramid']:
            bag_coords_scale = bag_coords
            patch_size = bag_info['patch_size']

            # Calculate ratio for reshaping pixel-level attention scores spatially
            print(patch_size)
            print(scale)
            ratio = int(-(-patch_size / scale if scale != 'aggregated' else patch_size / args.scales[0] // 1))
            attention_scores = attention_scores.reshape(len(bag_coords_scale), ratio, ratio)

        # Initialize empty tensors for accumulating attention values and counts
        attention_map = torch.zeros(img_h, img_w)
        attention_map_counts = torch.zeros(img_h, img_w)

        # Loop over each patch coordinate for the current scale
        for patch_idx in range(len(bag_coords_scale)):

            # Get x,y coordinates of the patch (top-left corner)
            x, y = bag_coords_scale[patch_idx, :]
            x = x.item()
            y = y.item()

            x_start = max(0, x)
            x_end = min(img_w, x + patch_size)
            y_start = max(0, y)
            y_end = min(img_h, y + patch_size)

            if args.multi_scale_model in ['fpn', 'backbone_pyramid']:

                # Upsample the spatial attention patch map to full patch size
                patch_map = F.interpolate(attention_scores[patch_idx].unsqueeze(0).unsqueeze(0),
                                          size=(patch_size, patch_size), mode='bilinear',
                                          align_corners=True).detach().cpu().squeeze()

                # Normalize patch_map to [0,1]
                patch_map = (patch_map - patch_map.min()) / (
                            patch_map.max() - patch_map.min() + torch.finfo(torch.float16).eps)

                # Add normalized patch attention to the aggregated attention map for this scale
                attention_map[y_start:y_end, x_start:x_end] += patch_map

            elif args.multi_scale_model == 'msp':
                # Directly add scalar patch-level attention score for this patch to the attention map region
                attention_map[y_start:y_end, x_start:x_end] += attention_scores[patch_idx]

            attention_map_counts[y_start:y_end, x_start:x_end] += 1

        # Compute average attention per pixel
        heatmap = torch.where(attention_map_counts == 0, torch.tensor(0.0),
                              torch.div(attention_map, attention_map_counts))

        # Apply Gaussian smoothing
        heatmap = torch.from_numpy(gaussian_filter(heatmap, sigma=10))

        # Normalize heatmap values only inside the segmentation mask, zero outside
        heatmap = torch.where(torch.tensor(seg_mask, dtype=torch.bool),
                              (heatmap - heatmap[seg_mask != 0].min()) / (
                                          heatmap[seg_mask != 0].max() - heatmap[seg_mask != 0].min()),
                              torch.tensor(0.0))

        # Extract bounding boxes from heatmap
        predicted_bboxes = extract_bounding_boxes_from_heatmap(heatmap,
                                                               quantile_threshold=args.quantile_threshold,
                                                               max_bboxes=args.max_bboxes,
                                                               min_area=args.min_area,
                                                               iou_threshold=args.iou_threshold)

        # Store heatmap and bounding boxes for each scale
        if args.type_scale_aggregator in ['concatenation', 'gated-attention']:

            heatmaps[scale] = {
                "heatmap": heatmap,
                "pred_bboxes": predicted_bboxes,
                "scale_score": scale_score
            }

            aggregated_heatmap += heatmap * scale_score


    # If aggregated heatmap is not already included in heatmaps dict
    if 'aggregated' not in heatmaps:
        # Normalize aggregated heatmap to [0,1]
        aggregated_heatmap = (aggregated_heatmap - aggregated_heatmap.min()) / (
                    aggregated_heatmap.max() - aggregated_heatmap.min())

        # Extract bounding boxes from aggregated heatmap
        predicted_bboxes = extract_bounding_boxes_from_heatmap(aggregated_heatmap,
                                                               quantile_threshold=args.quantile_threshold,
                                                               max_bboxes=args.max_bboxes, min_area=args.min_area,
                                                               iou_threshold=args.iou_threshold)

        # Add aggregated heatmap and bboxes to heatmaps dictionary
        heatmaps["aggregated"] = {
            "heatmap": aggregated_heatmap,
            "pred_bboxes": predicted_bboxes
        }

    return bag_prob, heatmaps

def main(args):
    # seed_all(args.seed)  # Fix the seed for reproducibility

    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nUsing device:', device)

    args.apex = True if args.apex == "y" else False

    args.running_interactive = True if args.running_interactive == "y" else False

    torch.cuda.empty_cache()  # Clean up

    if args.feature_extraction == 'online':
        if 'efficientnetv2' in args.arch:
            args.model_base_name = 'efficientv2_s'
        elif 'efficientnet_b5_ns' in args.arch:
            args.model_base_name = 'efficientnetb5'
        else:
            args.model_base_name = args.arch

    args.n_class = 1  # Binary classification task

    # Define class labels
    if args.label.lower() == 'mass':
        class0 = 'not_mass'
        class1 = 'mass'
    elif args.label.lower() == 'suspicious_calcification':
        class0 = 'not_calcification'
        class1 = 'calcification'

    label_dict = {class0: 0, class1: 1}

    # Build model and load model checkpoint
    if args.clip_chk_pt_path is None or not os.path.exists(args.clip_chk_pt_path):
        print('\nMammoCLIP checkpoint not found')
        model_config = 'b2-model-best-epoch-10'
        url = f"https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/{model_config}.tar?download=true"
        output_dir = "./checkpoints/"
        filename = os.path.join(output_dir, f"{model_config}.tar")
        # Ensure the checkpoints directory exists
        os.makedirs(output_dir, exist_ok=True)
        # download the file
        if not os.path.exists(filename):
            print(f"Downloading model from {url}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print(f"Download completed and saved to {filename}")
            else:
                print(f"Failed to download the model. Status code: {response.status_code}")
        # # extract the tar file
        # with tarfile.open(filename, "r:tar") as tar:
        #     tar.extractall(path=output_dir)
        print('model saved')
        args.clip_chk_pt_path = filename

    global model
    model = build_model(args)
    print('model loaded')
    checkpoint_path = os.path.join('checkpoints/', 'best_FPN-MIL_calcifications.pth')
    if not os.path.exists(checkpoint_path):
        os.makedirs('checkpoints/', exist_ok=True)
        gdown.download('https://drive.google.com/file/d/1pcr5wa8cI7R8L-7MfkXBEBB2IE02NmMI/view?usp=sharing', checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.is_training = False  # Set model mode for evaluation
    model.eval()

    ## Launch gradio demo
    # Build Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("## 🖼️ My Image Classifier")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload or Drag & Drop Image")

        with gr.Row():
            classify_button = gr.Button("Run Classifier")
            output_label = gr.Label(label="Result")
            output_image = gr.Image(label="Output")  # Add a new Image component to display

        classify_button.click(fn=run_classifier, inputs=image_input, outputs=[output_label, output_image])
    demo.launch()


# This function is called when the button is pressed
def run_classifier(image):
    if image is None:
        return "No image uploaded"

    # Load and preprocess image
    with torch.no_grad():
        tfm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(mean=args.mean, std=args.std),
                                              lambda_funct(pad_image, args.patch_size, args.mean, args.std),
                                              Patching(
                                                  patch_size=args.patch_size,
                                                  overlap=args.overlap,
                                                  multi_scale_model=args.multi_scale_model,
                                                  scales=args.scales
                                              )
                                              ])
        x, bag_coords, padding = tfm(image)
        padding_left, padding_right, padding_top, padding_bottom = padding
        print(padding)

        bag_info = {
            'patch_size': args.patch_size,
            'step_size': args.patch_size - int(args.patch_size * args.overlap[0]),
            'img_height': args.img_size[0] + padding_top + padding_bottom,
            'img_width': args.img_size[1] + padding_left + padding_right,
            # 'img_dir': img_path
        }

        # Process image
        model.to(device)
        output = model(x.unsqueeze(0).to(device))
        bag_prob = torch.sigmoid(output)

        # Visualize detected lesions
        vis = visualize_detection(args, model, image, bag_coords, bag_info)

        prob = bag_prob.cpu().detach().squeeze().numpy()
    print('prob:', prob)
    return {"no_calcification": 1-prob, "has_calcification": prob}




if __name__ == "__main__":
    args = config()
    main(args)
