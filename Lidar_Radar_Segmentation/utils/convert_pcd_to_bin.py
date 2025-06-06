import os
import numpy as np
import open3d as o3d
from tqdm import tqdm 

def convert_pcd_to_bin(pcd_dir, bin_dir):
    os.makedirs(bin_dir, exist_ok=True)
    for filename in tqdm(os.listdir(pcd_dir),"Converting PCD to BIN"):
        if filename.endswith(".pcd"):
            pcd_path = os.path.join(pcd_dir, filename)
            bin_path = os.path.join(bin_dir, filename.replace(".pcd", ".bin"))
            
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            if not pcd.has_points():
                print(f"Skipping empty point cloud: {filename}")
                continue

            # Add dummy intensity if needed
            if points.shape[1] == 3:
                intensities = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack((points, intensities))
            
            points.astype(np.float32).tofile(bin_path)
            # print(f"Converted: {filename} → {bin_path}")
