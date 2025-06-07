import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud
from numpy.linalg import svd
from scipy.spatial.transform import Rotation as R

# directory del dataset TruckScenes
TRUCKSCENES_ROOT = "Lidar_Radar_Segmentation\Dataset\man-truckscenes"
# versione del dataset TruckScenes (ad esempio "v1.0-mini")
VERSION = "v1.0-mini"
# sample da processare
SAMPLE_INDEX = 10


from sklearn.linear_model import RANSACRegressor


from mpl_toolkits.mplot3d import Axes3D

def plot_ground_plane(points, inlier_mask, model, num_grid=10):
    """
    Visualizza i punti con il piano stimato.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Punti del terreno
    ax.scatter(points[inlier_mask][:, 0], points[inlier_mask][:, 1], points[inlier_mask][:, 2],
               c='brown', s=1, label='Ground Points')

    # Punti rimanenti
    ax.scatter(points[~inlier_mask][:, 0], points[~inlier_mask][:, 1], points[~inlier_mask][:, 2],
               c='blue', s=1, label='Non-Ground Points')

    # Piano stimato (superficie)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], num_grid)
    y = np.linspace(ylim[0], ylim[1], num_grid)
    X, Y = np.meshgrid(x, y)
    Z = model.predict(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)

    ax.plot_surface(X, Y, Z, color='green', alpha=0.4, label='Estimated Ground Plane')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ground Plane Estimation (RANSAC)")
    ax.legend()
    plt.show()



def remove_ground_plane(points, threshold=0.2, max_z=1.5):
    """
    Rimuove i punti del terreno stimati con RANSAC.
    Returns:
        points_no_ground: punti senza terreno
        inlier_mask: maschera booleana dei punti classificati come terreno
        model: piano stimato (RANSAC)
    """
    candidate_points = points[points[:, 2] < max_z]

    if len(candidate_points) < 10:
        return points, np.zeros(len(points), dtype=bool), None

    X = candidate_points[:, :2]
    y = candidate_points[:, 2]

    model = RANSACRegressor(residual_threshold=threshold)
    model.fit(X, y)

    z_pred = model.predict(points[:, :2])
    dist = np.abs(points[:, 2] - z_pred)
    inlier_mask = dist < threshold

    return points[~inlier_mask], inlier_mask, model




def segmentation(points, threshold=0.7, min_samples=10):
    """
    Segmenta la nuvola di punti utilizzando DBSCAN.
    Argomenti:
        points (array numpy Nx3): Nuvola di punti 
            [3 è il numero delle dimensioni dei punti: x, y, x].
        threshold (float): Distanza massima tra due punti per essere considerati nello stesso cluster.
        min_samples (int): Numero minimo di punti per formare un cluster.
    Returns:
        segments (list): Lista di segmenti, ogni segmento è un array numpy con i punti che lo compongono.
    """
    # verifica che i punti siano in un formato corretto
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("I punti devono essere un array Nx3.")
    # calcola il clustering sui punti privati delle intensità
    clustering = DBSCAN(eps=threshold, min_samples=min_samples).fit(points)
    # ottieni le etichette dei cluster
    labels = clustering.labels_
    segments = []
    # per ogni etichetta unica, forma un segmento ed estrai i punti corrispondenti
    for label in np.unique(labels):
        if label == -1:
            continue
        segments.append(points[labels == label])
    return segments



def get_OBB(segment):
    """
    Calcola la Oriented Bounding Box (OBB) per un segmento di nuvola di punti.
    Argomenti:
        segment (array numpy Nx4): Segmento di nuvola di punti [x, y, z, intensity].
    Returns:
        dict: Dizionario con chiavi 'center' (array np con dim 3), 'size' (array np con dim 3), 
        'rotation' (array np con dim 4) che rappresentano il centro della box, 
        le dimensioni e la rotazione in formato quaternion.
    """
    # verifica che il segmento sia in un formato corretto
    if segment.ndim != 2 or segment.shape[1] != 3:
        raise ValueError("Il segmento deve essere un array Nx3.")
    # calcola il baricentro e centra i punti
    centroid = np.mean(segment, axis=0)
    centered = segment - centroid
    # esegue la PCA con SVD
    U, S, Vt = svd(centered)
    # ottiene la matrice di rotazione e ne corregge la direzione se è necessario
    rotation_matrix = Vt.T
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1
    # proietta i punti nel sistema ruotato
    projected = centered @ rotation_matrix
    # calcola gli angoli della box e le sue dimensioni in questo sistema
    min_corner = np.min(projected, axis=0)
    max_corner = np.max(projected, axis=0)
    size = max_corner - min_corner
    # calcola il centro della box nel sistema ruotato e poi lo trasforma in coordinate del sistema orginale
    box_center_rotated = (min_corner + max_corner) / 2.0
    box_center_original = centroid + box_center_rotated @ rotation_matrix.T
    # converte la matrice di rotazione in quaternion compatibile con il formato di TruckScenes
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    # [x, y, z, w] -> [w, x, y, z]
    quat_wxyz = np.roll(quat_xyzw, 1)

    return {
        "center": box_center_original,
        "size": size,                 
        "rotation": quat_wxyz         
    }



def draw_boxes_top_view(points, boxes, figsize=(10, 10)):
    plt.figure(figsize=figsize)

    # Scatter con altezza Z come colore, limitata tra -2 e +8 metri
    sc = plt.scatter(
        points[:, 0], points[:, 1],
        s=0.5,
        c=points[:, 2],
        cmap='viridis',
        vmin=-2, vmax=8,   # limiti fissi della scala colore
        alpha=0.8
    )
    plt.colorbar(sc, label='Altezza (Z) [m]')

    for box in boxes:
        center = box['center']
        size = box['size']
        quat = box['rotation']

        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rot_matrix = rot.as_matrix()

        l, w = size[0], size[1]
        dx = l / 2
        dy = w / 2

        corners = np.array([
            [ dx,  dy, 0],
            [ dx, -dy, 0],
            [-dx, -dy, 0],
            [-dx,  dy, 0],
        ])

        world_corners = (rot_matrix @ corners.T).T + center

        x = world_corners[:, 0]
        y = world_corners[:, 1]
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), 'r')

    plt.title("Top View with Height Heatmap (Z-axis Coloring, Limited to [-2m, +8m])")
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()



def voxel_downsample(points, voxel_size=0.1):
    """
    Esegue un filtraggio per voxelizzazione, mantenendo un solo punto per ciascun voxel.
    Utile per rimuovere duplicati o punti molto vicini.
    """
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]




def main():
    # === STEP 1: Inizializzazione del dataset ===
    trucksc = TruckScenes(version=VERSION, dataroot=TRUCKSCENES_ROOT, verbose=True)
    sample = trucksc.sample[SAMPLE_INDEX]
    sample_token = sample['token']
    lidar_tokens = {sensor_name: sample['data'][sensor_name] for sensor_name in sample['data'] if 'LIDAR' in sensor_name}

    all_points = []

    for sensor_name in lidar_tokens.keys():
        token = lidar_tokens[sensor_name]
        sd = trucksc.get('sample_data', token)
        filepath = os.path.join(trucksc.dataroot, sd['filename'])
        pc = LidarPointCloud.from_file(filepath)

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

        print(f"Processed {sensor_name} with {points.shape[1]} points.")

        all_points.append(points.T)  # (N, 3)

    fused_points = np.vstack(all_points)  # shape (N_total, 3)


    print(f"Total points after fusion: {fused_points.shape[0]}")

    z_filtered_points, ground_mask, ground_model = remove_ground_plane(fused_points, threshold=0.2, max_z=1.5)
    plot_ground_plane(fused_points, ground_mask, ground_model)
  
    print(f"Points after z-thresholding: {z_filtered_points.shape[0]}")
  
    downsampled_points = voxel_downsample(z_filtered_points, voxel_size=0.1)
  
    print(f"Points after voxel downsampling: {downsampled_points.shape[0]}")

  
    trucksc.render_sample(sample_token)

    segments = segmentation(downsampled_points)

    boxes = [get_OBB(seg) for seg in segments]

    draw_boxes_top_view(downsampled_points, boxes)



if __name__ == "__main__":
    main()
    
