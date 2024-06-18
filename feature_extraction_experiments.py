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
        'std_hue'        : np.std(h),
        'std_saturation' : np.std(s),
        # 'std_value'      : np.std(v) note: mungkin ini gaperlu
    }


'''
algoritma: 
    "menengahkan" dan "merotasi" gambar sehingga nilai asymmetry indexnya sekecil mungkin
    dgn nilai center dan angle dari cv2.fitEllipse(). Lalu bandingkan seberapa banyak yg
    "overlap" dibanding luas asli, terhadap sumbu x dan y. Lalu rata-rata hasilnya.
note:
    Definisi ini aku buat sendiri soalnya ngga nemu referensi yang menstandarkan perhitungan
    indeks simetri dari gambar. Dari definisi ini, harusnya sih nilainya dari 0-1 dimana
    1 itu perfect symmetry (horizontal dan vertikal)
'''
def calc_symmetry(segmented, contour):
    ellipse = cv2.fitEllipse(contour)
    (x, y), (MA, ma), angle = ellipse
    
    cx, cy = int(x), int(y)
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_image = cv2.warpAffine(segmented, rotation_matrix, (segmented.shape[1], segmented.shape[0]))
    centered =  _pad_center(rotated_image, (cx, cy))

    flip_v = cv2.flip(centered, 0)
    flip_h = cv2.flip(centered, 1)
    overlap_v = np.sum((centered > 0) & (flip_v > 0))
    overlap_h = np.sum((centered > 0) & (flip_h > 0))
    area = np.sum(centered > 0)
    
    return {
        'symmetry': (overlap_h + overlap_v) * 0.5 / area,
        'area': area / (segmented.shape[0] * segmented.shape[1])
        }


def get_features(dataset_path, image_id):
    segmented = cv2.cvtColor(cv2.imread(f'{dataset_path}/{image_id}_segmentation.png'), cv2.COLOR_BGR2GRAY)
    image = cv2.imread(f'{dataset_path}/{image_id}.jpg')

    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    features = {}

    features.update(calc_hsv(image, segmented))
    features.update(calc_symmetry(segmented, contour))

    return features