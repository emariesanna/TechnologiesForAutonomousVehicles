import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor



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