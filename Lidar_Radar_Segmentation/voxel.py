import numpy as np
from scipy.spatial.transform import Rotation as R


def voxelize_box(box, resolution=0.1):
    """
    Voxelizza un box 3D ruotato, restituendo un set di tuple voxel occupati.
    """
    center = box['center']
    size = box['size']
    quat = box['rotation']  # [x, y, z, w]

    # Crea griglia locale (in coordinate del box allineato)
    l, w, h = size
    x = np.arange(-l/2, l/2, resolution)
    y = np.arange(-w/2, w/2, resolution)
    z = np.arange(-h/2, h/2, resolution)
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)

    # Ruota e trasla i voxel nel sistema globale
    rot = R.from_quat(quat)
    grid_world = rot.apply(grid) + center

    # Converti a voxel grid discreta
    voxel_indices = np.floor(grid_world / resolution).astype(int)
    voxel_set = set(map(tuple, voxel_indices))

    return voxel_set


def voxel_downsample(points, voxel_size=0.1):
    """
    Esegue un filtraggio per voxelizzazione, mantenendo un solo punto per ciascun voxel.
    Utile per rimuovere duplicati o punti molto vicini.
    """
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]