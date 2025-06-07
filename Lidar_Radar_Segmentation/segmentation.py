import numpy as np
from sklearn.cluster import DBSCAN
from numpy.linalg import svd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


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
    print("Segmenting points with DBSCAN...")
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
        segment (array numpy Nx3): Segmento di nuvola di punti [x, y, z].
    Returns:
        dict: Dizionario con chiavi 'center' (array np con dim 3), 'size' (array np con dim 3), 
        'rotation' (array np con dim 4) che rappresentano il centro della box, 
        le dimensioni e la rotazione in formato quaternion [w, x, y, z].
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


def draw_boxes_top_view(points, boxes, gtboxes=None):
    plt.figure(figsize=(12, 8))

    # Calcolo dinamico dei limiti dell'altezza
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    sc = plt.scatter(
        points[:, 0], points[:, 1],
        s=0.5,
        c=points[:, 2],
        cmap='viridis',
        vmin=z_min, vmax=z_max,
        alpha=0.8
    )
    plt.colorbar(sc, label='Altezza (Z) [m]')

    def draw_single_box(box, color, linestyle='-'):
        center = box['center']
        size = box['size']
        quat = box['rotation']

        # Converti quaternion in formato [x, y, z, w] se necessario
        quat_xyzw = np.roll(quat, -1)  # [w, x, y, z] → [x, y, z, w]
        rot = R.from_quat(quat_xyzw)
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
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), color=color, linestyle=linestyle, linewidth=2)

    # Disegna box predette in rosso
    for box in boxes:
        draw_single_box(box, color='red')

    # Disegna box GT in blu tratteggiate se presenti
    if gtboxes is not None:
        for gt in gtboxes:
            draw_single_box(gt, color='blue', linestyle='--')

    plt.title("Top View: Predicted (Red) vs GT (Blue) - Z Height Colored")
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.tight_layout()
    plt.show()



