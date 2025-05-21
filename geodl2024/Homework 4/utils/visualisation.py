import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d import Axes3D



def generate_contour_(norm_func, xlim, ylim, grid_pts, scale):
    """Generate contour data for the given norm function."""
    grid_pts = [grid_pts[0] + 2, grid_pts[1]]
    xx = torch.linspace(xlim[0], xlim[1], grid_pts[0])
    xx = xx[1:-1]  # remove points close to the poles
    yy = torch.linspace(ylim[0], ylim[1], grid_pts[1])

    contour_data = []
    for x_center in xx:
        for y_center in yy:
            alpha = 2 * scale
            x_lim, y_lim = alpha * 1.0, alpha * 1 / torch.sqrt(torch.cos(x_center) ** 2)
            tensors = (
                torch.linspace(-x_lim, x_lim, 25),
                torch.linspace(-y_lim, y_lim, 25),
            )
            grid_x, grid_y = torch.meshgrid(*tensors, indexing="ij")

            x = torch.Tensor([x_center, y_center])
            grid = torch.stack(
                (grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1
            )  # shape (n_pts * n_pts, 2)
            x = x.repeat(grid.shape[0], 1)
            norm = norm_func(grid, x)
            norm = norm.reshape(grid_x.shape)

            contour_data.append((x_center, y_center, grid_x, grid_y, norm))
    return contour_data


def plot_contour(ax, norm_func, xlim=[-5, 5], ylim=[-5, 5], grid_pts=[10, 10], scale=0.1, color="r"):
    """Plot the indicatrix of the norm function on the given axis."""
    contour_data = generate_contour_(norm_func, xlim, ylim, grid_pts, scale)
    for x_center, y_center, grid_x, grid_y, norm in contour_data:
        figc, axc = plt.subplots(1, 1)
        cs = axc.contour(grid_x, grid_y, norm, levels=[scale], colors=color)
        plt.close(figc)
        polygon = cs.allsegs[0]

        if len(polygon) != 1:
            print("issue at ({},{}), contour broken".format(x_center, y_center))
            print("norm max: {}".format(np.max(norm)))
        else:
            pp = polygon[0] + [x_center, y_center]
            codes = Path.CURVE3 * np.ones(len(pp), dtype=Path.code_type)
            codes[0] = codes[-1] = Path.MOVETO
            path = Path(pp, codes)

            ax.add_patch(PathPatch(path, color=color, fill=True, lw=0, alpha=0.2))
        ax.scatter(x_center, y_center, color="k", s=1)
    return ax


def plot_contour_3d(ax, norm_func, xlim=[-np.pi/2, np.pi/2], ylim=[-np.pi/2, np.pi/2], grid_pts=[10, 10], scale=0.1, color="r", alpha=0.2):
    """Plot the indicatrix of the norm function on the given axis in 3D."""
    contour_data = generate_contour_(norm_func, xlim, ylim, grid_pts, scale)
    for x_center, y_center, grid_x, grid_y, norm in contour_data:
        figc, axc = plt.subplots(1, 1)
        cs = axc.contour(grid_x, grid_y, norm, levels=[scale], colors=color)
        plt.close(figc)
        polygon = cs.allsegs[0]

        if len(polygon) != 1:
            print("issue at ({},{}), contour broken".format(x_center, y_center))
            print("norm max: {}".format(np.max(norm)))
        else:
            pp = polygon[0] + [x_center, y_center]
            pp = np.array([map_to_sphere_np(p) for p in pp])

            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color=color, alpha=alpha)
            ax.scatter3D(*map_to_sphere_np([x_center, y_center]), color="k", s=1)
    return ax


def map_to_sphere_np(x):
    """Map a point on the sphere from spherical coordinates to Cartesian coordinates"""
    theta, phi = x
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.array([x, y, z])

def plot_sphere(ax, color="r"):
    """Plot the sphere for visualisation."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.1)
    return ax
