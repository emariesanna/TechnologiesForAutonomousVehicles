import json
import os
import pickle
import numpy as np
import random
from tqdm import tqdm

def quaternion_yaw(q):
    """Convert quaternion to yaw (w, x, y, z)"""
    import math
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def build_infos(
    annotation_path,
    sample_data_path,
    lidar_dir,
    radar_dir,
    output_dir,
    split_ratio=0.8
):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    with open(sample_data_path, 'r') as f:
        sample_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Organizza per sample_token
    anns_by_token = {}
    for ann in annotations:
        token = ann["sample_token"]
        if token not in anns_by_token:
            anns_by_token[token] = []
        anns_by_token[token].append(ann)

    # Trova i file key_frame
    lidar_samples = {}
    radar_samples = {}
    for entry in sample_data:
        if not entry["is_key_frame"]:
            continue

        sample_token = entry["sample_token"]
        filename = entry["filename"]
        timestamp = entry["timestamp"]

        if "LIDAR_LEFT" in filename and sample_token in anns_by_token:
            lidar_samples[sample_token] = {
                "path": os.path.join(lidar_dir, os.path.basename(filename)).replace(".pcd", ".bin"),
                "timestamp": timestamp
            }
        elif "RADAR_LEFT_FRONT" in filename and sample_token in anns_by_token:
            radar_samples[sample_token] = {
                "path": os.path.join(radar_dir, os.path.basename(filename)).replace(".pcd", ".bin"),
                "timestamp": timestamp
            }

    # Prendi solo i token che esistono sia per lidar che radar
    common_tokens = list(set(lidar_samples.keys()) & set(radar_samples.keys()))
    common_tokens.sort()
    random.seed(42)
    random.shuffle(common_tokens)

    split_idx = int(len(common_tokens) * split_ratio)
    train_tokens = common_tokens[:split_idx]
    val_tokens = common_tokens[split_idx:]

    def generate_info_entries(tokens, sensor_samples, sensor_type):
        infos = []
        for token in tqdm(tokens, desc=f"Building {sensor_type} infos"):
            anns = anns_by_token[token]
            sample = sensor_samples[token]
            boxes = []
            names = []

            for ann in anns:
                x, y, z = ann["translation"]
                w, l, h = ann["size"]
                qw, qx, qy, qz = ann["rotation"]
                yaw = quaternion_yaw([qw, qx, qy, qz])

                boxes.append([x, y, z, w, l, h, yaw])
                names.append("vehicle")  # oppure: ann["category_name"]

            infos.append({
                'lidar_path': sample["path"] if sensor_type == "lidar" else None,
                'radar_path': sample["path"] if sensor_type == "radar" else None,
                'gt_boxes': np.array(boxes, dtype=np.float32),
                'gt_names': names,
                'token': token,
                'timestamp': sample["timestamp"]
            })

        return infos

    # Genera .pkl
    lidar_train = generate_info_entries(train_tokens, lidar_samples, "lidar")
    lidar_val = generate_info_entries(val_tokens, lidar_samples, "lidar")
    radar_train = generate_info_entries(train_tokens, radar_samples, "radar")
    radar_val = generate_info_entries(val_tokens, radar_samples, "radar")

    with open(os.path.join(output_dir, 'truckscenes_lidar_infos_train.pkl'), 'wb') as f:
        pickle.dump(lidar_train, f)
    with open(os.path.join(output_dir, 'truckscenes_lidar_infos_val.pkl'), 'wb') as f:
        pickle.dump(lidar_val, f)
    with open(os.path.join(output_dir, 'truckscenes_radar_infos_train.pkl'), 'wb') as f:
        pickle.dump(radar_train, f)
    with open(os.path.join(output_dir, 'truckscenes_radar_infos_val.pkl'), 'wb') as f:
        pickle.dump(radar_val, f)

    # Genera split file
    imagesets_dir = os.path.join(output_dir, 'ImageSets')
    os.makedirs(imagesets_dir, exist_ok=True)
    with open(os.path.join(imagesets_dir, 'train.txt'), 'w') as f:
        for token in train_tokens:
            f.write(token + '\n')
    with open(os.path.join(imagesets_dir, 'val.txt'), 'w') as f:
        for token in val_tokens:
            f.write(token + '\n')

    print(f"Totale sample: {len(common_tokens)} | Train: {len(train_tokens)} | Val: {len(val_tokens)}")
