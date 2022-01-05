import numpy as np
from PIL import Image
from icevision.all import *

def IoU(box1, box2, mode='xyxy'):
    """ Calculate the IoU of a box pair
    box1: (xmin1, ymin1, width1, height1)
    box2: (xmin2, ymin2, width2, height2)
    """
    if mode == 'xywh':
        # Extract params
        xmin1, ymin1, width1, height1 = box1
        xmin2, ymin2, width2, height2 = box2
        
        # Compute the x, y maxes
        xmax1 = xmin1 + width1 - 1
        xmax2 = xmin2 + width2 - 1
        
        ymax1 = ymin1 + height1 - 1
        ymax2 = ymin2 + height2 - 1

    elif mode == 'xyxy':
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

    else:
        raise Exception(f"Unsupported mode type '{mode}'")
    
    xA = min(xmax1, xmax2)
    xB = max(xmin1, xmin2)
    x_overlap = max(xA - xB + 1, 0)
    
    yA = min(ymax1, ymax2)
    yB = max(ymin1, ymin2)
    y_overlap = max(yA - yB + 1, 0)
    
    intersection = x_overlap * y_overlap
    
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    # area1 = width1 * height1
    # area2 = width2 * height2
    
    union = area1 + area2 - intersection
    
    iou = intersection / union
    
    return iou

def calculate_conf(gts, preds, confs, mode='xyxy'):
    confs_order = np.argsort(confs)[::-1]
    preds = preds[confs_order]

    main_ioumat = np.zeros((len(gts), len(preds)))

    for i, gt in enumerate(gts):
        for j, pred in enumerate(preds):
            main_ioumat[i, j] = IoU(gt, pred, mode=mode)

    # We'll need this at each threshold
    preds_set = set([x for x in range(0, len(preds))])

    tps = 0
    fps = 0
    fns = 0

    for thr in np.arange(0.3, 0.85, 0.05):
    #     print(thr)

        ioumat = main_ioumat.copy()
        ioumat[ioumat < thr] = 0
        # ioumat

        mask = (ioumat != 0)
        res = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)

        tp_set = set(res[res != -1])

        tps += len(tp_set)
        fps += len(preds_set - tp_set)
        fns += len(res) - len(tp_set)

    return tps, fps, fns

def calculate_conf_all_samples(gts, bboxes, scores, mode='xyxy'):
    total_tps = 0
    total_fps = 0
    total_fns = 0

    for i, _ in enumerate(gts):
        
        # If the image has no bboxes, and we predicted no bboxes, no accumulation
        if (len(gts[i]) == 0) and (len(bboxes[i]) == 0):
            tps, fps, fns = 0, 0, 0
            
        # If the image has no bboxes, and we predicted bboxes, all are false positives
        elif (len(gts[i]) == 0) and (len(bboxes[i]) != 0):
            tps, fps, fns = 0, len(bboxes[i]), 0
            
        # If the image has bboxes, and we predicted no bboxes, all are false negatives
        elif (len(gts[i]) != 0) and (len(bboxes[i]) == 0):
            tps, fps, fns = 0, 0, len(gts[i])
            
        # Otherwise let's do the full calculation
        else:
            tps, fps, fns = calculate_conf(gts[i], bboxes[i], scores[i], mode=mode)
            
        total_tps += tps
        total_fps += fps
        total_fns += fns

    return total_tps, total_fps, total_fns

def compute_f2(model_type,
               model,
               valid_ds,
               valid_tfms,
               class_map,
               mode='xyxy',
               thr_min=0.1,
               thr_max=0.9,
               thr_step=0.05):
    ### Get predictions from the trained model
    detection_threshold = 0.1
    preds = []

    # Make prediction in the original image coordinates
    # Note: iterating through valid_ds.records is siginificantly
    # faster than iterating through valid_ds
    for entry in tqdm(valid_ds.records):
        img = Image.open(entry.filepath)
        
        pred = model_type.end2end_detect(img, 
                                        valid_tfms, 
                                        model, 
                                        class_map=class_map, 
                                        detection_threshold=detection_threshold)
        preds.append(pred)
        
        img.close()

    thrs = np.arange(thr_min, thr_max + thr_step, thr_step)
    f2s = []

    for thr in thrs:
    #     print(thr)
        # thr = 0.5

        ### Extract the relevant parameters from predictions
        bboxes = []
        scores = []

        for pred in preds:
            current_bboxes = pred['detection']['bboxes']
            current_scores = pred['detection']['scores']

            # Filter based on current threshold value thr
            output_bboxes, output_scores = [], []
            for i, _ in enumerate(current_bboxes):
                if current_scores[i] > thr:
                    output_bboxes.append(current_bboxes[i].xyxy)
                    output_scores.append(current_scores[i])

            bboxes.append(np.array(output_bboxes))
            scores.append(np.array(output_scores))

        ### Get the ground truth boxes

        gts = []

        for record in valid_ds.records:
            gts.append(np.array([bb.xyxy for bb in record.detection.bboxes]))

        # Calculate dataset-wide tps, fps, fns
        total_tps, total_fps, total_fns = calculate_conf_all_samples(gts, bboxes, scores, mode=mode)

        # Compute precision, recall, F2
        if (total_tps + total_fps) > 0:
            precision = total_tps / (total_tps + total_fps)
        else:
            precision = 0
        if (total_tps + total_fns) > 0:
            recall = total_tps / (total_tps + total_fns)
        else:
            recall = 0
            
        if (precision != 0) and (recall != 0):
            f2 = 5 * precision * recall / (4 * precision + recall)
        else:
            f2 = 0
        
        f2s.append(f2)
    
    return thrs, f2s