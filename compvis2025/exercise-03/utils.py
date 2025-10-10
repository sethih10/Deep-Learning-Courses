# Description:
#   Exercise3 utils.py.
#
# Copyright (C) 2018 Santiago Cortes, Juha Ylioinas
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

from __future__ import division

import numpy as np
from types import *
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
import ast

# convert from rgb to grayscale image
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2125 * r + 0.7154 * g + 0.0721 * b

    return gray

# salt-and-pepper noise generator
def imnoise(img, mode, prob):
    imgn = img.copy()
    if mode == 'salt & pepper':
        assert (prob >= 0 and prob <= 1), "prob must be a scalar between 0 and 1"
        h, w = imgn.shape
        prob_sp = np.random.rand(h, w)
        imgn[prob_sp < prob] = 0
        imgn[prob_sp > 1 - prob] = 1

    return imgn

# Gaussian noise generator
def add_gaussian_noise(img, noise_sigma):
    temp_img = np.copy(img)
    h, w = temp_img.shape
    noise = np.random.randn(h, w) * noise_sigma
    noisy_img = temp_img + noise
    return noisy_img

# 2d Gaussian filter
def gaussian2(sigma, N=None):

    if N is None:
        N = 2*np.maximum(4, np.ceil(6*sigma))+1

    k = (N - 1) / 2.

    xv, yv = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))

    # 2D gaussian filter    
    g = 1/(2 * np.pi * sigma**2) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))

    # 1st order derivatives
    gx = -xv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))
    gy = -yv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2)) 

    # 2nd order derivatives
    gxx = (-1 + xv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2*sigma**2)) / (2 * np.pi * sigma**4)
    gyy = (-1 + yv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2*sigma**2)) / (2 * np.pi * sigma**4)
    gxy = (xv * yv) / (2 * np.pi * sigma**6) * np.exp(-(xv**2 + yv**2) / (2*sigma**2))    

    return g, gx, gy, gxx, gyy, gxy

# fit an affine model between two 2d point sets
def affinefit(x, y):
    # Ordinary least squares (check wikipedia for further details):
    # 
    # Y                          = P*X_aug            % X_aug is in homogenous coords (one sample per col)
    # Y'                         = X_aug'*P'          % take transpose from both sides
    # X_aug*Y'                   = X_aug*X_aug'*P'    % multiply both sides from left by X_aug
    # inv(X_aug*X_aug')*X_aug*Y' = P'                 % multiply both sides from left by the inverse of X_aug*X_aug' 
    n = x.shape[0]
    x = x.T
    y = y.T
    x_aug = np.concatenate((x, np.ones((1, n))), axis=0)
    y_aug = np.concatenate((y, np.ones((1, n))), axis=0)
    xtx = np.dot(x_aug, x_aug.T)
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_x = np.dot(xtx_inv, x_aug)   
    P = np.dot(xtx_inv_x, y_aug.T)  
    A = P.T[0:2, 0:2]
    b = P.T[0:2, 2]

    return A, b

def contains_forbidden_usage(code: str, forbidden_names=()) -> bool:
    """
    Returns True if any forbidden function call or assignment is found in the code.

    Parameters:
    - code: The source code string to check.
    - forbidden_names: A tuple of names (function names, variable names, or attribute names) to look for.

    Supports detection of:
    - Direct function calls (e.g., conv2(...))
    - Attribute function calls (e.g., scipy.ndimage.convolve(...))
    - Aliased calls (e.g., from scipy.ndimage import convolve as conv3 → conv3(...))
    - Assignments from forbidden variable names
    """
    alias_map = {}

    try:
        tree = ast.parse(code)

        # First pass: track imported aliases
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                    full_attr = f"{module}.{alias.name}"
                    if alias.asname:
                        alias_map[alias.asname] = full_attr
                    else:
                        alias_map[alias.name] = full_attr

        # Second pass: detect usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func

                # Direct call
                if isinstance(func, ast.Name):
                    if func.id in forbidden_names:
                        return True
                    # Check alias resolution
                    if func.id in alias_map and alias_map[func.id] in forbidden_names:
                        return True

                # Attribute call
                elif isinstance(func, ast.Attribute):
                    if func.attr in forbidden_names:
                        return True
                    full_attr = f"{ast.unparse(func.value)}.{func.attr}"
                    if full_attr in forbidden_names:
                        return True

            # Assignments from forbidden variables
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                value = node.value
                if isinstance(value, ast.Name) and value.id in forbidden_names:
                    return True

        return False

    except SyntaxError:
        return False  # If the cell couldn't be parsed, assume it's okay


def show_pyramid_horizontally(pyramid, title="Pyramid", base_fig_width=12, base_fig_height=8, spacing=1, spacing2=75):
    fig = plt.figure(figsize=(base_fig_width, base_fig_height))
    plt.suptitle(title, fontsize=16)

    # Get total width in pixels for normalization
    widths = [pyramid[i].shape[1] for i in pyramid]
    heights = [pyramid[i].shape[0] for i in pyramid]
    total_width = sum(widths) + spacing * (len(pyramid) - 1)
    max_height = max(heights)

    x_offset = 0  # in pixel space

    for i in pyramid:
        img = pyramid[i]
        h, w = img.shape[:2]

        # Normalized width and height (relative to figure size)
        width_norm = w / total_width
        height_norm = h / max_height

        # Convert x_offset to normalized coordinates
        x_norm = x_offset / total_width
        y_norm = (1 - height_norm) / 2  # center vertically

        # Add axis for this image
        ax = fig.add_axes([x_norm, y_norm, width_norm, height_norm])
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Level {i}", fontsize=10)
        #ax.axis('off')

        # Update offset (in pixels)
        x_offset += w + spacing2

    plt.show()
    