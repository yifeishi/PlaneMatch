import numpy as np
import sys
import os
from numba import jit, double

sys.path.append(os.getcwd()) 

@jit
def mul(mat, vec):
    m00, m01, m02, m03 = double(mat[0][0]), double(mat[0][1]), double(mat[0][2]), double(mat[0][3])
    m10, m11, m12, m13 = double(mat[1][0]), double(mat[1][1]), double(mat[1][2]), double(mat[1][3])
    m20, m21, m22, m23 = double(mat[2][0]), double(mat[2][1]), double(mat[2][2]), double(mat[2][3])
    m30, m31, m32, m33 = double(mat[3][0]), double(mat[3][1]), double(mat[3][2]), double(mat[3][3])
    v0, v1, v2, v3 = double(vec[0]), double(vec[1]), double(vec[2]), double(vec[3])
    return np.array([m00 * v0 + m01 * v1 + m02 * v2 + m03 * v3,
                     m10 * v0 + m11 * v1 + m12 * v2 + m13 * v3,
                     m20 * v0 + m21 * v1 + m22 * v2 + m23 * v3,
                     m30 * v0 + m31 * v1 + m32 * v2 + m33 * v3])


@jit
def rgbd_cal_dmask2cbox(depth_img, col_img_h, col_img_w, dcm, ccm, ext, depth_scl, depth_mask):
    """
    Calibrating depth and color image for computing a bounding box
    in color image space based on the input depth mask
    """
    depth_height = depth_img.shape[0]
    depth_width = depth_img.shape[1]
    color_height = col_img_h
    color_width = col_img_w
    depth_mask_height = depth_mask.shape[0]
    depth_mask_width = depth_mask.shape[1]
    #assert depth_mask.shape == depth_img.shape, 'assertion failed -- rgbd_cal_dmask2cbox: depth_mask.shape == depth_img.shape'

    aligned = np.zeros((depth_height, depth_width, 3), dtype=np.float_)
    color_bbox = [color_width, color_height, 0, 0]

    fx_d, fy_d, cx_d, cy_d = double(dcm[0][0]), double(dcm[1][1]), double(dcm[0][2]), double(dcm[1][2])
    for v in range(depth_height):
        for u in range(depth_width):
            z = double(depth_img[v][u]) / double(depth_scl)
            x = (u - cx_d) * z / fx_d
            y = (v - cy_d) * z / fy_d
            transformed = mul(ext, np.array([x, y, z, 1]))
            aligned[v][u] = transformed[0:3]
    
    fx_c, fy_c, cx_c, cy_c = double(ccm[0][0]), double(ccm[1][1]), double(ccm[0][2]), double(ccm[1][2])
    validFlag = False
    for v in range(depth_mask_height):
        for u in range(depth_mask_width):
            if depth_mask[v][u] == 0 or aligned[v][u][2] == 0:
                continue
            x = aligned[v][u][0] * fx_c / aligned[v][u][2] + cx_c
            y = aligned[v][u][1] * fy_c / aligned[v][u][2] + cy_c
            if x > color_width or y > color_height or x < 0 or y < 0:
                continue
            color_bbox[0] = min(int(round(x)), color_bbox[0])
            color_bbox[1] = min(int(round(y)), color_bbox[1])
            color_bbox[2] = max(int(round(x)), color_bbox[2])
            color_bbox[3] = max(int(round(y)), color_bbox[3])
            validFlag = True
    if not validFlag:
        color_bbox = [0, 0, color_width, color_height]
    if color_bbox[2] - color_bbox[0] < 2:
        color_bbox[2] = color_bbox[0] + 2
    if color_bbox[3] - color_bbox[1] < 2:
        color_bbox[3] = color_bbox[1] + 2
    return (color_bbox)


@jit
def rgbd_calibration(depth_img, color_img, dcm, ccm, ext, depth_scl):
    depth_height = depth_img.shape[0]
    depth_width  = depth_img.shape[1]
    color_height = color_img.shape[0]
    color_width  = color_img.shape[1]

    point_coord = np.zeros((depth_height, depth_width, 3), dtype=np.float_)
    aligned_rgb = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)

    fx_d, fy_d, cx_d, cy_d = dcm[0][0], dcm[1][1], dcm[0][2], dcm[1][2]
    for v in range(depth_height):
        for u in range(depth_width):
            z = depth_img[v][u] / depth_scl
            x = (u - cx_d) * z / fx_d
            y = (v - cy_d) * z / fy_d
            transformed = mul(ext, [x, y, z, 1.])
            point_coord[v][u] = transformed[0:3]

    fx_c, fy_c, cx_c, cy_c = ccm[0][0], ccm[1][1], ccm[0][2], ccm[1][2]
    for v in range(depth_height):
        for u in range(depth_width):
            if point_coord[v][u][2] == 0:
                continue
            x = point_coord[v][u][0] * fx_c / point_coord[v][u][2] + cx_c
            y = point_coord[v][u][1] * fy_c / point_coord[v][u][2] + cy_c
            if x > color_width or y > color_height or x < 0 or y < 0:
                continue
            aligned_rgb[v][u] = color_img[int(round(y))][int(round(x))]

    return aligned_rgb