from tqdm import tqdm
from truckscenes import TruckScenes
from dataset import get_points_from_sample
from ground import remove_ground_plane, plot_ground_plane
from segmentation import segmentation, get_OBB, draw_boxes_top_view
from evaluate import get_ground_truth_boxes, analyze_box_matches
from voxel import voxel_downsample
import logging
import random

# directory del dataset TruckScenes
TRUCKSCENES_ROOT = "Lidar_Radar_Segmentation\Dataset\man-truckscenes"
# versione del dataset TruckScenes (ad esempio "v1.0-mini")
VERSION = "v1.0-mini"
# numero di sample da processare (se si vuole processare un sample specifico, impostare a 1)
NUM_SAMPLE = 1
# sample singolo da processare
SAMPLE_INDEX = 10
# imposta a True per considerare solo il primo sample di ogni scena (i sample sono divisi in scene, ciascuna scena ha 20 sample)
PER_SCENE = True
# sensori da considerare
SENSORS = ['LIDAR_LEFT', 'LIDAR_REAR', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT'
           'RADAR_LEFT_BACK', 'RADAR_LEFT_FRONT', 'RADAR_LEFT_SIDE', 'RADAR_RIGHT_BACK', 'RADAR_RIGHT_FRONT', 'RADAR_RIGHT_SIDE']
PRINT_GROUND_PLANE = True  # se True, stampa il ground plane
PRINT_SEGMENTATION = True  # se True, stampa la visualizzazione delle box previste
PRINT_VISUALIZATION = False  # se True, stampa la visualizzazione del sample


def main():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("log.txt", mode='w'),   # Log su file  
            # logging.StreamHandler()                   # Log su console
        ]
    )

    trucksc = TruckScenes(version=VERSION, dataroot=TRUCKSCENES_ROOT, verbose=False)

    all_sample_indices = []

    if PER_SCENE:
        if NUM_SAMPLE > len(trucksc.scene):
            raise ValueError(f"NUM_SAMPLE non può essere maggiore del numero di scene: ({len(trucksc.scene)})")
        elif NUM_SAMPLE > 0:
            for scene in trucksc.scene:
                sample_token = scene['first_sample_token']
                sample_indices = [i for i, s in enumerate(trucksc.sample) if s['token'] == sample_token]
                if len(sample_indices) != 1:
                    raise ValueError(f"Expected exactly one sample index for token {sample_token}, got {len(sample_indices)}")
                sample_index = sample_indices[0]
                all_sample_indices.append(sample_index)
        else:
            raise ValueError("NUM_SAMPLE deve essere maggiore di 0.")
    else:
        if NUM_SAMPLE > len(all_sample_indices):
            raise ValueError(f"NUM_SAMPLE non può essere maggiore della dimensione del dataset: ({len(all_sample_indices)})")
        elif NUM_SAMPLE > 0:
            all_sample_indices = list(range(len(trucksc.sample)))
        else:
            raise ValueError("NUM_SAMPLE deve essere maggiore di 0.")
    
    if NUM_SAMPLE > 1:
        random.shuffle(all_sample_indices)
        sample_indices = all_sample_indices[:NUM_SAMPLE]
    else:
        sample_indices = [SAMPLE_INDEX]
    
    total_good_matches = 0
    total_missed = 0
    total_poor_matches = 0
    total_over_segmented = 0
    total_under_segmented = 0
    
    for sample_index in tqdm(sample_indices, desc="Processing samples", unit="sample"):

        logging.info(f"Sample {sample_index}:")
        sample = trucksc.sample[sample_index]
        points = get_points_from_sample(trucksc, sample, SENSORS)
        logging.info(f"Total points after fusion: {points.shape[0]}")

        z_filtered_points, ground_mask, ground_model = remove_ground_plane(points, threshold=0.3, max_z=1.5)
        if PRINT_GROUND_PLANE:
            plot_ground_plane(points, ground_mask, ground_model)
        logging.info(f"Points after z-thresholding: {z_filtered_points.shape[0]}")
  
        downsampled_points = voxel_downsample(z_filtered_points, voxel_size=0.1)
        logging.info(f"Points after voxel downsampling: {downsampled_points.shape[0]}")

        gt_boxes = get_ground_truth_boxes(trucksc, sample)
        if PRINT_VISUALIZATION:
            sample_token = sample['token']
            trucksc.render_sample(sample_token)

        segments = segmentation(downsampled_points, threshold=0.5, min_samples=10)
        pred_boxes = [get_OBB(seg) for seg in segments]
        logging.info(f"Number of predicted boxes: {len(pred_boxes)}")
        logging.info(f"Number of ground truth boxes: {len(gt_boxes)}")

        if PRINT_SEGMENTATION:
            draw_boxes_top_view(downsampled_points, pred_boxes, gt_boxes)

        results = analyze_box_matches(gt_boxes, pred_boxes)
        total_good_matches += results['good_matches']
        total_missed += results['missed']
        total_poor_matches += results['poor_matches']
        total_over_segmented += results['over_segmented']
        total_under_segmented += results['under_segmented']
        
        logging.info(f"Good matches: {results['good_matches']}")
        logging.info(f"Missed: {results['missed']}")
        logging.info(f"Poor matches: {results['poor_matches']}")
        logging.info(f"Over-segmented: {results['over_segmented']}")
        logging.info(f"Under-segmented: {results['under_segmented']}")
    
    with open("Lidar_Radar_Segmentation/results.txt", "a") as f:
        f.write("Results:\n")
        f.write(f"Total Samples Processed: {len(sample_indices)}\n")
        f.write(f"Total Good Matches: {total_good_matches}\n")
        f.write(f"Total Missed: {total_missed}\n")
        f.write(f"Total Poor Matches: {total_poor_matches}\n")
        f.write(f"Total Over-segmented: {total_over_segmented}\n")
        f.write(f"Total Under-segmented: {total_under_segmented}\n")

if __name__ == "__main__":
    main()
    
