import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import numpy as np
from models.centernet3d import CenterNet3D  # aggiorna path se diverso
from spconv.pytorch.utils import PointToVoxel
from easydict import EasyDict as edict

def load_point_cloud(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

def voxelize(point_cloud, voxel_size, coors_range, max_points=5):
    from collections import defaultdict

    voxel_dict = defaultdict(list)

    # quantizzazione
    grid_size = ((coors_range[3:] - coors_range[:3]) / voxel_size).astype(int)

    for pt in point_cloud:
        x, y, z = pt[:3]
        ix = int((x - coors_range[0]) / voxel_size[0])
        iy = int((y - coors_range[1]) / voxel_size[1])
        iz = int((z - coors_range[2]) / voxel_size[2])

        if ix < 0 or ix >= grid_size[0] or iy < 0 or iy >= grid_size[1] or iz < 0 or iz >= grid_size[2]:
            continue

        voxel_dict[(0, iz, iy, ix)].append(pt)

    voxel_features = []
    voxel_coords = []

    for k, pts in voxel_dict.items():
        pts = np.array(pts)
        if len(pts) > max_points:
            pts = pts[:max_points]
        # media dei punti
        mean_feat = np.mean(pts, axis=0)
        voxel_features.append(mean_feat)
        voxel_coords.append(k)

    return np.array(voxel_features), np.array(voxel_coords, dtype=np.int32)

def preprocess(pc_np):
    voxel_size = np.array([0.2, 0.2, 0.4])  # (x, y, z)
    point_cloud_range = np.array([0, -40, -3, 70.4, 40, 1])  # es. KITTI
    voxels, coords = voxelize(pc_np, voxel_size, point_cloud_range)
    voxel_tensor = torch.tensor(voxels, dtype=torch.float32)
    coord_tensor = torch.tensor(coords, dtype=torch.int32)
    return voxel_tensor, coord_tensor


def load_model():
    configs = edict()
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2
    configs.num_conners = 4

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cenoff': configs.num_center_offset,
        'direction': configs.num_direction,
        'z': configs.num_z,
        'dim': configs.num_dim,
        'hm_conners': configs.num_classes
    }

    configs.head_conv = 64
    configs.sparse_shape = (40, 1600, 1400)

    model = CenterNet3D(configs.sparse_shape, configs.heads, configs.head_conv)
    model.cuda()
    model.eval()
    return model

def run_inference(model, voxel_features, voxel_coords, batch_size):
    
    voxel_features = voxel_features.to("cuda")
    voxel_coords = voxel_coords.to("cuda")
    model = model.to("cuda")
    print("Model on:", next(model.parameters()).device)
    print("voxel_features on:", voxel_features.device)
    print("voxel_coords on:", voxel_coords.device)
    print("Any NaN in features:", torch.isnan(voxel_features).any())
    print("Any NaN in coords:", torch.isnan(voxel_coords).any())
    import os
    os.environ["SPCONV_DEBUG_SAVE_PATH"] = "./spconv_debug"



    with torch.no_grad():
        output = model(voxel_features, voxel_coords, batch_size)
    return output



if __name__ == "__main__":

    path = "Lidar_Radar_Segmentation/Dataset/man-truckscenes/lidar_bin/LIDAR_LEFT_1692868171688148.bin"
    print(f"Loading point cloud from {path}")
    pc = load_point_cloud(path)
    print(f"Point cloud shape: {pc.shape}")
    if pc.shape[0] == 0:
        raise ValueError("Point cloud is empty. Please check the file path or content.")
    print("Preprocessing point cloud...")
    voxels, coords = preprocess(pc)
    print(f"Voxels shape: {voxels.shape}, Coords shape: {coords.shape}")
    if voxels.shape[0] == 0:
        raise ValueError("No voxels generated. Please check the voxelization parameters.")

    batch_size = 1
    print("Loading model...")
    model = load_model()
    print("Running inference...")
    
    try:
        results = run_inference(model, voxels, coords, batch_size)
        print(">>> Inference completata.")
        print(">>> Tipo risultati:", type(results))
        
        if isinstance(results, dict):
            print("Predicted heads:", results.keys())
            for k, v in results.items():
                print(f"{k}: {v.shape}")
        elif isinstance(results, torch.Tensor):
            print("Predicted tensor output:", results.shape)
        elif results is None:
            print(">>> Attenzione: results è None")
        else:
            print(">>> Tipo non gestito:", results)
    except Exception as e:
        print(">>> Errore durante l'inferenza:", e)
