# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import matplotlib.pyplot as plt
from datasets.coco import *
import cv2
from eval_metrics import *
from util.box_ops import box_cxcywh_to_xyxy

# from yolov7_utils.experimental import attempt_load
# from yolov7_utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
#     box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
# from yolov7_utils.metrics import ap_per_class, ConfusionMatrix
# from yolov7_utils.plots import plot_images, output_to_target, plot_study_txt
# from yolov7_utils.torch_utils import select_device, time_synchronized, TracedModel


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox.cpu())
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933], [0.2, 0.2, 0.2]]

CLASSES = ('Bipolar Forceps' ,'Prograsp Forceps', 'Large Needle Driver', 'Vessel Sealer',
            'Grasping Retractor', 'Monopolar Curved Scissors', 'Others', '') # For visualization, classes should add null as the last label
font = cv2.FONT_HERSHEY_SIMPLEX

def tensor_to_pil_img(img):
    # Process the original image after transformation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inv_std = 1 / std
    inv_mean = -mean / std
    denormalized_img = img * inv_std[:, None, None] + inv_mean[:, None, None]
    
    denormalized_img = np.array(denormalized_img.permute(1, 2, 0))
    denormalized_img = (denormalized_img - denormalized_img.min()) / (denormalized_img.max()-denormalized_img.min())
    visual_img = denormalized_img.copy()
    return visual_img

def plot_results_gt(img, prob, boxes, target, epoch, img_id, output_dir):
    save_dir = os.path.join(output_dir, 'imgs')
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    print('Save results:')
    
    pil_img = tensor_to_pil_img(img)

    # show prediction result(ax1) and ground truth(ax2) side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10)) 
    ax1.imshow(pil_img)
    ax2.imshow(pil_img)
    colors = COLORS * 100
    
    # draw prediction result
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        ax1.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=colors[cl], linewidth=3))

        text = f'{cl}{CLASSES[cl]}: {p[cl]:0.2f}'
        ax1.text(xmin, ymin, text, fontsize=12,
                bbox=dict(facecolor=colors[cl], alpha=0.5))
    
    # convert ground truth target to normal format
    t_boxes = rescale_bboxes(target['boxes'].cpu(), np.array(target['size'].cpu())[::-1])
    t_labels = target['labels'].cpu()

    # draw ground truth
    for cl, (xmin, ymin, xmax, ymax) in zip(t_labels, t_boxes.tolist()):
        ax2.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=colors[cl], linewidth=3))

        text = f'{cl}{CLASSES[cl]}'
        ax2.text(xmin, ymin, text, fontsize=12,
                bbox=dict(facecolor=colors[cl], alpha=0.5))
        
    ax1.set_title('Prediction Results')
    ax2.set_title('Ground Truth')
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig(os.path.join(save_dir, f'res{epoch}_{int(img_id):04d}.png'), bbox_inches='tight')
    print(f'Save the result image res{epoch}_{int(img_id):04d}.png')
    plt.show()

def plot_results(img, prob, boxes, epoch, img_id):
    save_dir = './outputs/predictions'
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    print('Save results:')
    img = tensor_to_pil_img(img)
    
    
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS):
        top_corner, down_corner = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        
        cv2.rectangle(img, top_corner, down_corner, color=c, thickness=4)
        
        cl = p.argmax() # prob has a shape of (8,) =num_classes
        text = f'{cl}:{CLASSES[cl]}: {p[cl]:0.2f}'
        cv2.putText(img,str(text),(top_corner[0]+5, top_corner[1]+25), font, 1,c,4,cv2.LINE_AA)
    
    plt.imshow(img)
    plt.imsave(os.path.join(save_dir, f'res{epoch}_{int(img_id):04d}.png'), img)
    print(f'Save the result image res{epoch}_{int(img_id):04d}.png')
    plt.show()

def plot_gt(img, boxes, classes, epoch, img_id):
    '''
    Args:
    img: from the dataloader
    boxes: from the dataloader
    '''
    save_dir = './outputs/gt'
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    # Denormalized the boxes
    h, w = img.shape[-2:]
    boxes *= torch.tensor([w, h, w, h], dtype=torch.float32)
    boxes = box_cxcywh_to_xyxy(boxes)

    visual_img = tensor_to_pil_img(img)
    
    # Draw and save the image with bounding boxes
    for cl, (xmin, ymin, xmax, ymax), c in zip(classes, boxes.tolist(), COLORS):
        top_corner, down_corner = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(visual_img, top_corner, down_corner, color=c, thickness=4)

        text = f'{cl}:{CLASSES[cl]}'
        cv2.putText(visual_img,str(text),(top_corner[0]+5, top_corner[1]+25), font, 1,c,4,cv2.LINE_AA)
    
    plt.imshow(visual_img)
    plt.show()
    plt.imsave(os.path.join(save_dir, f'gt_{int(img_id):04d}.png'), visual_img)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    train_freq = 0
    mode = 'train'

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device) # batch size: 2
        
        # print(samples.tensors.shape)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
   
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    val_freq = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        val_freq += 1

        outputs = model(samples) # batch size is 2
        
        # keep only predictions with 0.3+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.3

        # Choose the image in the first batch to visualize
        im_id = targets[0]['image_id']
        size = targets[0]['size']
        img = samples.tensors[0].cpu()

        # orig_img, _ = data_loader.dataset[im_id-1800]

        # convert boxes from [0; 1] to image scales
        output_boxes = rescale_bboxes(outputs['pred_boxes'][0, keep], np.array(size.cpu())[::-1])
        target = targets[0]

        if val_freq % 10 == 0:
            plot_results_gt(img, probas[keep], output_boxes, target, epoch, im_id, output_dir)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # convert raw bbox => xyxy and mutiply by scales [img_w, img_h, img_w, img_h]
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
