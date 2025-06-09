import os
import numpy as np
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from scipy.spatial.transform import Rotation as R
import logging

def get_points_from_sample(dataset, sample, sensors):
    
    data_tokens = {data_sensor: sample['data'][data_sensor] for data_sensor in sample['data'] if data_sensor in sensors}

    all_points = []

    for data_sensor in data_tokens.keys():
        token = data_tokens[data_sensor]
        sd = dataset.get('sample_data', token)
        filepath = os.path.join(dataset.dataroot, sd['filename'])
        
        if 'LIDAR' in data_sensor:
            pc = LidarPointCloud.from_file(filepath)
        elif 'RADAR' in data_sensor:
            pc = RadarPointCloud.from_file(filepath)
        else:
            raise ValueError(f"Unsupported sensor type: {data_sensor}")

        points = pc.points[:3, :]
        cs_record = dataset.get('calibrated_sensor', sd['calibrated_sensor_token'])
        quat_wxyz = cs_record['rotation']
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        rotation = R.from_quat(quat_xyzw).as_matrix()
        translation = np.array(cs_record['translation'])

        points = pc.points[:3, :]
        points = (rotation @ points) + translation.reshape(3, 1)

        logging.info(f"Processed {data_sensor} with {points.shape[1]} points.")

        all_points.append(points.T)

    fused_points = np.vstack(all_points)
    return fused_points