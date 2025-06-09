from tqdm import tqdm
import numpy as np
from voxel import voxelize_box
from scipy.spatial.transform import Rotation as R


def iou_3d_voxelized(box1, box2, resolution=0.1):
    voxels1 = voxelize_box(box1, resolution)
    voxels2 = voxelize_box(box2, resolution)

    inter = voxels1 & voxels2
    union = voxels1 | voxels2

    if not union:
        return 0.0
    return len(inter) / len(union)


def analyze_box_matches(gt_boxes, pred_boxes, iou_fn=iou_3d_voxelized, good_iou=0.25, tollerance_iou=0.05, max_center_distance=5.0):
    
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    iou_matrix = np.zeros((n_gt, n_pred))

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            dist = np.linalg.norm(np.array(gt['center']) - np.array(pred['center']))
            if dist > max_center_distance:
                iou_matrix[i, j] = -1
            else:
                iou_matrix[i, j] = iou_fn(pred, gt)

    good_matches = 0
    missed = 0
    poor_matches = 0
    over_segmented = 0
    under_segmented = 0

    per_gt_analysis = []
    per_pred_analysis = []

    for i in range(n_gt):
        ious = iou_matrix[i]
        max_iou = np.max(ious)
        count_overlapping = np.sum(ious > tollerance_iou)

        if max_iou > good_iou:
            good_matches += 1
            label = ['good']
        elif max_iou > 0:
            poor_matches += 1
            label = ['poor']
        else:
            missed += 1
            label = ['missed']

        if count_overlapping > 1:
            over_segmented += 1
            label.append('over_segmented')
        
        per_gt_analysis.append({
            'gt_index': i,
            'max_iou': max_iou,
            'n_pred_overlapping': count_overlapping,
            'label': label
        })

    for j in range(n_pred):
        ious = iou_matrix[:, j]
        count_gt = np.sum(ious > tollerance_iou)

        if count_gt > 1:
            under_segmented += 1
            label = 'under_segmented'
        elif count_gt == 1:
            label = 'matched'
        else:
            label = 'unmatched'

        per_pred_analysis.append({
            'pred_index': j,
            'max_iou': np.max(ious),
            'n_gt_matched': count_gt,
            'label': label
        })

    return {
        'good_matches': good_matches,
        'missed': missed,
        'poor_matches': poor_matches,
        'over_segmented': over_segmented,
        'under_segmented': under_segmented,
        'per_gt_analysis': per_gt_analysis,
        'per_pred_analysis': per_pred_analysis
    }


def get_ground_truth_boxes(trucksc, sample):
    gt_boxes = []

    sample_data_token = next(iter(sample['data'].values()))
    sample_data = trucksc.get('sample_data', sample_data_token)

    ego_pose = trucksc.get('ego_pose', sample_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])   
    ego_rotation = np.array(ego_pose['rotation'])         
    ego_rot = R.from_quat(np.roll(ego_rotation, -1))      
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
