import os
import numpy as np
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from scipy.spatial.transform import Rotation as R
from ground import remove_ground_plane, plot_ground_plane
from segmentation import segmentation, get_OBB, draw_boxes_top_view
from evaluate import evaluate, get_ground_truth_boxes
from voxel import voxel_downsample

# directory del dataset TruckScenes
TRUCKSCENES_ROOT = "Lidar_Radar_Segmentation\Dataset\man-truckscenes"
# versione del dataset TruckScenes (ad esempio "v1.0-mini")
VERSION = "v1.0-mini"
# sample da processare
SAMPLE_INDEX = 10
# sensori da considerare
SENSORS = ['LIDAR_LEFT', 'LIDAR_REAR', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT'
           'RADAR_LEFT_BACK', 'RADAR_LEFT_FRONT', 'RADAR_LEFT_SIDE', 'RADAR_RIGHT_BACK', 'RADAR_RIGHT_FRONT', 'RADAR_RIGHT_SIDE']


def main():
    # === STEP 1: Inizializzazione del dataset ===
    trucksc = TruckScenes(version=VERSION, dataroot=TRUCKSCENES_ROOT, verbose=True)
    sample = trucksc.sample[SAMPLE_INDEX]
    data_tokens = {data_sensor: sample['data'][data_sensor] for data_sensor in sample['data'] if data_sensor in SENSORS}

    all_points = []

    for data_sensor in data_tokens.keys():
        token = data_tokens[data_sensor]
        sd = trucksc.get('sample_data', token)
        filepath = os.path.join(trucksc.dataroot, sd['filename'])
        
        
        if 'LIDAR' in data_sensor:
            pc = LidarPointCloud.from_file(filepath)
        elif 'RADAR' in data_sensor:
            pc = RadarPointCloud.from_file(filepath)
        else:
            raise ValueError(f"Unsupported sensor type: {data_sensor}")

        points = pc.points[:3, :]
        # Applica la trasformazione nel sistema del veicolo (ego frame)
        cs_record = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        # TruckScenes (come nuScenes) fornisce il quaternion come [w, x, y, z]
        quat_wxyz = cs_record['rotation']
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]  # Converti in [x, y, z, w]
        rotation = R.from_quat(quat_xyzw).as_matrix()
        translation = np.array(cs_record['translation'])

        # Trasforma i punti dal frame del sensore al frame ego vehicle
        points = pc.points[:3, :]  # shape (3, N)
        points = (rotation @ points) + translation.reshape(3, 1)

        print(f"Processed {data_sensor} with {points.shape[1]} points.")

        all_points.append(points.T)  # (N, 3)

    fused_points = np.vstack(all_points)  # shape (N_total, 3)


    print(f"Total points after fusion: {fused_points.shape[0]}")


    z_filtered_points, ground_mask, ground_model = remove_ground_plane(fused_points, threshold=0.2, max_z=1.5)
    # plot_ground_plane(fused_points, ground_mask, ground_model)
  
    print(f"Points after z-thresholding: {z_filtered_points.shape[0]}")
  
    downsampled_points = voxel_downsample(z_filtered_points, voxel_size=0.05)
  
    print(f"Points after voxel downsampling: {downsampled_points.shape[0]}")

    gt_boxes = get_ground_truth_boxes(trucksc, sample)
    sample_token = sample['token']
    # trucksc.render_sample(sample_token)

    segments = segmentation(downsampled_points, threshold=0.6, min_samples=20)

    boxes = [get_OBB(seg) for seg in segments]

    draw_boxes_top_view(downsampled_points, boxes, gt_boxes)

    results = evaluate(pred_boxes=boxes, gt_boxes=gt_boxes, iou_threshold=0.25)
    print(results)



if __name__ == "__main__":
    main()
    
