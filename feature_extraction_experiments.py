# Author  : martabakmenari 
# NOTE: masi pake chatGPT dikit, tapi kebanyakan aku sendiri yg nulis,
#       soalnya GPT banyak ngawurnya.

import cv2
import numpy as np

def _pad_center(image, center):
    x, y   = image.shape
    cx, cy = center
    dx, dy = (x/2) - cx, (y/2) - cy
    pad_x = (int(dx), 0) if dx > 0 else (0, -int(dx))
    pad_y = (int(dy), 0) if dy > 0 else (0, -int(dy))
    return np.pad(image, (pad_x, pad_y), mode='constant', constant_values=0)


'''
algoritma:
    mengambil value "hue" dan "saturation" untuk setiap pixel dari image
    yang ada di dalam segmented, kemudian menghitung stdev nya.
'''
def calc_hsv(image, segmented_image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=segmented_image)
    
    hsv_flatten = masked_image.reshape((-1, 3))
    hsv = hsv_flatten[~(hsv_flatten == np.zeros(3)).all(axis=1)]

    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    return {
        'std_color' : np.std(s) + np.std(h),
        # 'std_value'      : np.std(v) note: mungkin ini gaperlu
    }


def calc_symmetry(segmented_image):
    M = cv2.moments(segmented_image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    top_left = segmented_image[:cY, :cX]
    top_right = segmented_image[:cY, cX:]
    bottom_left = segmented_image[cY:, :cX]
    bottom_right = segmented_image[cY:, cX:]
    
    symmetry_score = (cv2.matchShapes(top_left, bottom_right, 1, 0.0) + 
                      cv2.matchShapes(top_right, bottom_left, 1, 0.0)) / 2
    return { 'symmetry': symmetry_score }

def calc_reciprocal_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    return { '1/circularity': perimeter**2 / (4 * np.pi * area) }

def calc_smoothness(contour):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]

    from scipy.ndimage import gaussian_filter1d
    x_smooth = gaussian_filter1d(x, sigma=1)
    y_smooth = gaussian_filter1d(y, sigma=1)

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
    return { 'smoothness': 1 / np.var(curvature) }

def calc_std_color(image, segmented_image):
    masked_image = cv2.bitwise_and(image, image, mask=segmented_image)
    color_stddev = np.std(masked_image[segmented_image > 0], axis=0)
    return { 'std_color': np.mean(color_stddev) }

def get_features(dataset_path, image_id):
    segmented = cv2.cvtColor(cv2.imread(f'{dataset_path}/{image_id}_segmentation.png'), cv2.COLOR_BGR2GRAY)
    image = cv2.imread(f'{dataset_path}/{image_id}.jpg')

    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    features = {}

    # features.update(calc_hsv(image, segmented))
    features.update(calc_symmetry(segmented))
    features.update(calc_std_color(image, segmented))
    features.update(calc_reciprocal_circularity(contour))
    features.update(calc_smoothness(contour))

    return features