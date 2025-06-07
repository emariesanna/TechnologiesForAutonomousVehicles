from tqdm import tqdm
import numpy as np
from voxel import voxelize_box
from scipy.spatial.transform import Rotation as R

def evaluate(pred_boxes, gt_boxes, iou_threshold=0.25):
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in tqdm(pred_boxes, desc="Evaluating Predictions"):
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = iou_3d_voxelized(pred, gt, resolution=0.1)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        # print(f"Prediction: {pred['center']}, Best GT Index: {best_gt_idx}, Best IoU: {best_iou:.4f}")

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def iou_3d_voxelized(box1, box2, resolution=0.1):
    voxels1 = voxelize_box(box1, resolution)
    voxels2 = voxelize_box(box2, resolution)

    inter = voxels1 & voxels2
    union = voxels1 | voxels2

    if not union:
        return 0.0
    return len(inter) / len(union)



def get_ground_truth_boxes(trucksc, sample):
    gt_boxes = []

    sample_data_token = next(iter(sample['data'].values()))
    sample_data = trucksc.get('sample_data', sample_data_token)

    ego_pose = trucksc.get('ego_pose', sample_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])      # [x, y, z]
    ego_rotation = np.array(ego_pose['rotation'])            # [w, x, y, z]
    ego_rot = R.from_quat(np.roll(ego_rotation, -1))         # [x, y, z, w]
    ego_rot_inv = ego_rot.inv()

    correction = R.from_euler('z', -90, degrees=True)

    for ann_token in sample['anns']:
        ann = trucksc.get('sample_annotation', ann_token)

        center_world = np.array(ann['translation'])
        center_ego = ego_rot_inv.apply(center_world - ego_translation)

        ann_rot = R.from_quat(np.roll(ann['rotation'], -1))
        rot_ego = ego_rot_inv * ann_rot

        rot_ego_corrected = correction * rot_ego
        quat_ego = np.roll(rot_ego_corrected.as_quat(), 1)

        gt_boxes.append({
            'center': center_ego,
            'size': np.array(ann['size']),
            'rotation': quat_ego,
            'category': ann['category_name']
        })

    return gt_boxes
