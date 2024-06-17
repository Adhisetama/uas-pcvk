import cv2
import numpy as np

def calculate_hsv(image, segmented_image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=segmented_image)
    
    # hsv_segmented = hsv_image[mask]
    hsv_flatten = masked_image.reshape((-1, 3))
    hsv = hsv_flatten[~(hsv_flatten == np.zeros(3)).all(axis=1)]

    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    return {
        'std_hue'        : np.std(h),
        'std_saturation' : np.std(s),
        'std_value'      : np.std(v)
    }

def get_features(dataset_path, image_id):
    segmented = cv2.cvtColor(cv2.imread(f'{dataset_path}/{image_id}_segmentation.png'), cv2.COLOR_BGR2GRAY)
    image = cv2.imread(f'{dataset_path}/{image_id}.jpg')

    features = {}

    features.update(calculate_hsv(image, segmented))

    return features