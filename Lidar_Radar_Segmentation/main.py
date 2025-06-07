from truckscenes import TruckScenes
from utils.build_infos import build_infos
from utils.convert_pcd_to_bin import convert_pcd_to_bin

def main():
    annotation_path = 'Lidar_Radar_Segmentation/Dataset/man-truckscenes/v1.0-mini/sample_annotation.json'
    sample_data_path = 'Lidar_Radar_Segmentation/Dataset/man-truckscenes/v1.0-mini/sample_data.json'
    lidar_dir = 'Lidar_Radar_Segmentation/Dataset/man-truckscenes/samples/LIDAR_LEFT'
    lidar_bin_dir = 'Lidar_Radar_Segmentation/Dataset/man-truckscenes/lidar_bin'
    radar_dir = 'Lidar_Radar_Segmentation/Dataset/man-truckscenes/samples/RADAR_LEFT_FRONT'
    radar_bin_dir = 'Lidar_Radar_Segmentation/Dataset/man-truckscenes/radar_bin'
    output_path = 'Lidar_Radar_Segmentation/Dataset/man-truckscenes'

    convert_pcd_to_bin(lidar_dir, lidar_bin_dir)
    convert_pcd_to_bin(radar_dir, radar_bin_dir)    


    build_infos(annotation_path, sample_data_path, lidar_bin_dir, radar_bin_dir, output_path)

if __name__ == '__main__':
    main()
