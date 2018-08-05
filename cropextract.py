import os
import sys
import cv2
import PIL.Image as Image
import numpy as np
from scipy import ndimage
from rgbd_cal import rgbd_cal_dmask2cbox

grey_color = (128, 128, 128)
black_color = (0, 0, 0)

class SegInfo:
    def __init__(self, minx=0, miny=0, maxx=0, maxy=0, width=1, height=1):
        self.minx = width
        self.miny = height
        self.maxx = 0
        self.maxy = 0
        self.numpnt = 0
        self.mask = np.zeros((height, width), np.uint8)

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax, rmax
	
def GetSegmentsInfo(filename, seg_id):
    seg_info = SegInfo()
    pixels = np.asarray(Image.open(filename).convert('L'), dtype=np.uint8)
    masked = pixels - np.uint8(seg_id)
    _,seg_info.mask = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV)
    seg_info.numpnt = np.count_nonzero(seg_info.mask)
    seg_info.minx, seg_info.miny, seg_info.maxx, seg_info.maxy = bbox2(seg_info.mask)
    return seg_info

def CreateContextMaskHollow_OCV(img):
    rv_img = 255-img
    dt = ndimage.distance_transform_edt(rv_img)
    tmax = np.amax(dt)
    scl = 255./tmax
    dt *= scl
    dt = 255-dt
    return dt-img

def CreateContextMask_OCV(img):
    rv_img = 255-img
    dt = ndimage.distance_transform_edt(rv_img)
    tmax = np.amax(dt)
    scl = 255./tmax
    dt *= scl
    dt = 255-dt
    return dt

def CreateCroppedImage_OCV(img, left, upper, right, lower, fill_col, scl_fac=1., truncate=False):
    img_xmin = 0
    img_ymin = 0
    img_xmax = img.shape[1]-1
    img_ymax = img.shape[0]-1

    crop_centx = left + (right - left) / 2.
    crop_centy = lower + (upper - lower) / 2.
    crop_inner_size = round(max((right - left), (lower - upper)))
    crop_size = crop_inner_size * scl_fac
    crop_hsize = round(crop_size / 2.)
    crop_xmin = round(crop_centx - crop_hsize)
    crop_xmax = crop_xmin + crop_size
    crop_ymin = round(crop_centy - crop_hsize)
    crop_ymax = crop_ymin + crop_size

    cross_boundary = False
    cross_x = cross_y = 0
    if crop_xmin < img_xmin:
        crop_xmin = img_xmin
        cross_boundary = True
        cross_x = -1
    if crop_ymin < img_ymin:
        crop_ymin = img_ymin
        cross_boundary = True
        cross_y = -1
    if crop_xmax > img_xmax:
        crop_xmax = img_xmax
        cross_boundary = True
        cross_x = 1
    if crop_ymax > img_ymax:
        crop_ymax = img_ymax
        cross_boundary = True
        cross_y = 1

    if not cross_boundary:
        #return img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        return img[int(crop_ymin):int(crop_ymax), int(crop_xmin):int(crop_xmax)]
    else:
        x_offset = 0
        y_offset = 0
        if truncate:
            # The region beyong the boundary will be truncated, since we now
            #   compute a new crop size based on the truncated extents
            crop_size_x = crop_xmax - crop_xmin
            crop_size_y = crop_ymax - crop_ymin
            if crop_size_x < crop_size_y and cross_x == 1:
                x_offset = crop_size_y - crop_size_x
            if crop_size_x > crop_size_y and cross_y == 1:
                y_offset = crop_size_x - crop_size_y
            crop_size = max(crop_size_x, crop_size_y)
        else:
            # The region beyong the boundary is not truncated by filling with grey
            #   since we are using the desired crop size
            if cross_x == -1:
                x_offset = img_xmin - crop_xmin
            if cross_y == -1:
                y_offset = img_ymin - crop_ymin

        if len(img.shape) == 2:
            rslt_image = np.zeros((int(crop_size), int(crop_size)), img.dtype)
        else: # must be greater than 2
            rslt_image = np.zeros((int(crop_size), int(crop_size), int(3)), img.dtype)
        rslt_image[:,:] = fill_col
        crop_img = img[int(crop_ymin):int(crop_ymax), int(crop_xmin):int(crop_xmax)]
        rslt_image[int(y_offset):int(y_offset+crop_img.shape[0]), int(x_offset):int(x_offset+crop_img.shape[1])] = crop_img
        return rslt_image

def cropextract(col_fn, depth_fn, normal_fn, camera_path, seg_fn, seg_id):
    # Frame name
    fn = col_fn[col_fn.rfind('/')+1:len(col_fn)]
    # Open color image
    col_img = cv2.imread(col_fn)
    # Open depth image
    #depth_img = cv2.imread(depth_fn, 0)
    depth_img = Image.open(depth_fn).convert('I')
    depth_img = np.asarray(depth_img)
    # Open normal map
    normal_img = cv2.imread(normal_fn)
    # Open and get segmentation info.
    seg = GetSegmentsInfo(seg_fn, seg_id)
    # Calibration
    rgbd_calibration = False
    color_intr_path = os.path.join(camera_path, 'COLOR_INTRINSICS')
    if os.path.isfile(color_intr_path):
        color_intr = np.loadtxt(color_intr_path, usecols=range(3))
#        print('Using RGB camera intrinsics.')
    color_extr_path = os.path.join(camera_path, 'COLOR_EXTRINSICS')
    if os.path.isfile(color_extr_path):
        color_extr = np.loadtxt(color_extr_path, usecols=range(4))
#        print('Using RGB camera extrinsics.')
    depth_intr_path = os.path.join(camera_path, 'DEPTH_INTRINSICS')
    if os.path.isfile(depth_intr_path):
        depth_intr = np.loadtxt(depth_intr_path, usecols=range(3))
#        print('Using depth camera intrinsics.')
    depth_extr_path = os.path.join(camera_path, 'DEPTH_EXTRINSICS')
    if os.path.isfile(depth_extr_path):
        depth_extr = np.loadtxt(depth_extr_path, usecols=range(4))
#        print('Using depth camera extrinsics.')
    if os.path.isfile(color_intr_path) and os.path.isfile(color_extr_path) and os.path.isfile(depth_intr_path) and os.path.isfile(depth_extr_path):
        rgbd_calibration = True

    verbose = False
    # Crop color image
    if rgbd_calibration:
        rgb_bbox = rgbd_cal_dmask2cbox(depth_img, col_img.shape[0], col_img.shape[1], depth_intr, color_intr, depth_extr, 1000, seg.mask)
        img_crop = CreateCroppedImage_OCV(col_img, rgb_bbox[0], rgb_bbox[1], rgb_bbox[2], rgb_bbox[3], fill_col=grey_color)   # local crop
    else:
        img_crop = CreateCroppedImage_OCV(col_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=grey_color)   # local crop
    
    img_crop = cv2.resize(img_crop, (224, 224), interpolation = cv2.INTER_CUBIC)
    rgb_local_image = img_crop
    #######################################################
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.jpg')]+'_seg_{}_l_color'.format(seg_id)+'.png'
        cv2.imwrite(os.path.join('./results', img_crop_fn), img_crop)
    #######################################################
    if rgbd_calibration:
        img_crop = CreateCroppedImage_OCV(col_img, rgb_bbox[0], rgb_bbox[1], rgb_bbox[2], rgb_bbox[3], fill_col=grey_color, scl_fac=5, truncate=True)   # local crop
    else:
        img_crop = CreateCroppedImage_OCV(col_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=grey_color, scl_fac=5, truncate=True)    # global crop
    
    img_crop = cv2.resize(img_crop, (224, 224), interpolation = cv2.INTER_CUBIC)
    rgb_global_image = img_crop
    #######################################################
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.jpg')]+'_seg_{}_g_color'.format(seg_id)+'.png'
        cv2.imwrite(os.path.join('./results', img_crop_fn), img_crop)
    #######################################################

    # Crop depth image
    img_crop = CreateCroppedImage_OCV(depth_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0)   # local crop
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.png')]+'_seg_{}_l_depth'.format(seg_id)+'.png'
        f = open('./depth_b_max.txt','a')
        f.write('{} depth_max: {} depth_min: {} \n'.format(img_crop_fn, img_crop.max(), img_crop.min()))
        f.close()
    # use PIL for single channel resizing
    img_crop = Image.fromarray(img_crop, mode='I').resize((224,224), resample=Image.BILINEAR)
    depth_local_image = np.asarray(img_crop, dtype=np.int32)
    #######################################################
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.png')]+'_seg_{}_l_depth'.format(seg_id)+'.png'
        img_crop.save(os.path.join('./results', img_crop_fn))
        f = open('./depth_max.txt','a')
        f.write('{} depth_max: {} depth_min: {} \n'.format(img_crop_fn, depth_local_image.max(), depth_local_image.min()))
        f.close()
    #######################################################
    img_crop = CreateCroppedImage_OCV(depth_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0, scl_fac=5, truncate=True)    # global crop
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.png')]+'_seg_{}_l_depth'.format(seg_id)+'.png'
        f = open('./depth_b_max.txt','a')
        f.write('{} depth_max: {} depth_min: {} \n'.format(img_crop_fn, img_crop.max(), img_crop.min()))
        f.close()
    # use PIL for single channel resizing
    img_crop = Image.fromarray(img_crop, mode='I').resize((224,224), resample=Image.BILINEAR)
    depth_global_image = np.asarray(img_crop, dtype=np.int32)
    #######################################################
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.png')]+'_seg_{}_g_depth'.format(seg_id)+'.png'
        img_crop.save(os.path.join('./results', img_crop_fn))
        f = open('./depth_max.txt','a')
        f.write('{} depth_max: {} depth_min: {} \n'.format(img_crop_fn, depth_global_image.max(), depth_global_image.min()))
        f.close()
    #######################################################
    
    # Crop normal map
    img_crop = CreateCroppedImage_OCV(normal_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=black_color)   # local crop
    img_crop = cv2.resize(img_crop, (224, 224), interpolation = cv2.INTER_CUBIC)
    normal_local_image = img_crop
    #######################################################
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.jpg')]+'_seg_{}_l_normal'.format(seg_id)+'.png'
        cv2.imwrite(os.path.join('./results', img_crop_fn), img_crop)
    #######################################################
    img_crop = CreateCroppedImage_OCV(normal_img, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=black_color, scl_fac=5, truncate=True)    # global crop
    img_crop = cv2.resize(img_crop, (224, 224), interpolation = cv2.INTER_CUBIC)
    normal_global_image = img_crop
    #######################################################
    if verbose:
        img_crop_fn = fn[0:len(fn)-len('.jpg')]+'_seg_{}_g_normal'.format(seg_id)+'.png'
        cv2.imwrite(os.path.join('./results', img_crop_fn), img_crop)
    #######################################################
    
    # Crop segment mask image
    img_mask_crop = CreateCroppedImage_OCV(seg.mask, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0)   # local crop
    img_mask_crop = cv2.resize(img_mask_crop, (224, 224), interpolation = cv2.INTER_CUBIC)
    mask_local_image = img_mask_crop
    #######################################################
    if verbose:
        img_mask_crop_fn = fn[0:len(fn)-len('.jpg')]+'_seg_{}_l_mask'.format(seg_id)+'.png'
        cv2.imwrite(os.path.join('./results', img_mask_crop_fn), img_mask_crop)
    #######################################################
    img_mask_cntx = CreateContextMask_OCV(img_mask_crop)
    mask_local_c_image = img_mask_cntx
    #######################################################
    img_mask_crop = CreateCroppedImage_OCV(seg.mask, seg.minx, seg.miny, seg.maxx, seg.maxy, fill_col=0, scl_fac=5, truncate=True)    # global crop
    img_mask_crop = cv2.resize(img_mask_crop, (224, 224), interpolation = cv2.INTER_CUBIC)
    mask_global_image = img_mask_crop
    #######################################################
    if verbose:
        img_mask_crop_fn = fn[0:len(fn)-len('.jpg')]+'_seg_{}_g_mask'.format(seg_id)+'.png'
        cv2.imwrite(os.path.join('./results', img_mask_crop_fn), img_mask_crop)
    #######################################################
    img_mask_cntx = CreateContextMask_OCV(img_mask_crop)
    mask_global_c_image = img_mask_cntx
    #######################################################
    if verbose:
        img_mask_cntx_fn = fn[0:len(fn)-len('.jpg')]+'_seg_{}_g_mask_c'.format(seg_id)+'.png'
        cv2.imwrite(os.path.join('./results', img_mask_cntx_fn), img_mask_cntx)
    return (rgb_global_image, depth_global_image, normal_global_image, mask_global_c_image,
        rgb_local_image, depth_local_image, normal_local_image, mask_local_image)
