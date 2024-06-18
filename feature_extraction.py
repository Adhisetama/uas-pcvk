# author    : martabakmenari
# corresp   : chat.openai.com
# NOTE: isi file ini 99% copas dari chatGPT yh

import cv2
import numpy as np

def calculate_asymmetry(segmented_image):
    # Calculate the centroid
    M = cv2.moments(segmented_image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Split the image into four quadrants
    top_left = segmented_image[:cY, :cX]
    top_right = segmented_image[:cY, cX:]
    bottom_left = segmented_image[cY:, :cX]
    bottom_right = segmented_image[cY:, cX:]
    
    # Calculate the asymmetry index
    symmetry_score = (cv2.matchShapes(top_left, bottom_right, 1, 0.0) + 
                      cv2.matchShapes(top_right, bottom_left, 1, 0.0)) / 2
    return symmetry_score

def calculate_border_irregularity(segmented_image):
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    border_irregularity = perimeter**2 / (4 * np.pi * area)
    return border_irregularity

def calculate_color_irregularity(image, segmented_image):
    masked_image = cv2.bitwise_and(image, image, mask=segmented_image)
    color_stddev = np.std(masked_image[segmented_image > 0], axis=0)
    color_irregularity = np.mean(color_stddev)
    return color_irregularity

def calculate_differential_structure(segmented_image):
    # Use connected components or blob detection for structural analysis
    num_labels, labels_im = cv2.connectedComponents(segmented_image)
    sizes = [np.sum(labels_im == label) for label in range(1, num_labels)]
    average_size = np.mean(sizes)
    differential_structure = len(sizes) / average_size
    return differential_structure

def get_features(dataset_path, image_id):
    segmented = cv2.cvtColor(cv2.imread(f'{dataset_path}/{image_id}_segmentation.png'), cv2.COLOR_BGR2GRAY)
    image = cv2.imread(f'{dataset_path}/{image_id}.jpg')

    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    return {
        'asymmetry': calculate_asymmetry(segmented),
        'border_irregularity': calculate_border_irregularity(segmented),
        'color_irregularity': calculate_color_irregularity(image, segmented),
        'differential_structure': calculate_differential_structure(segmented)
    }