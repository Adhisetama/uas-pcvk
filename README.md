#### **Model Comparison**
Comparison of all models shown in table below:

| Model              | Precision | Accuracy | Recall  | F1 Score |
|--------------------|-----------|----------|---------|----------|
| Logistic Regression| 0.70      | 0.70     | 0.70    | 0.70     |
| SVM                | 0.65      | 0.65     | 0.65    | 0.65     |
| KNN                | 0.68      | 0.68     | 0.68    | 0.67     |
| MLP                | 0.70      | 0.68     | 0.68    | 0.67     |

#### **Feature Extractions**
the feature needed to be extracted from image is
1. `symmetry`
    > Calculated by splitting the segmented image into four with the center is centroid of the lesions. then match the crossing region (top-left to down-right; top-right to down-left) with `cv2.matchShapes()`.
2. `std_color`
    > Done by calculating all standard deviation for every color channel (RGB) in the image then averaging the values. The image must be segmented and masked by the segmentation.
3. `1/circularity`
    > Done by calculating circularity (4pi*area)/perimeter^2 but the numerator and denominator is flipped

#### **Model Training**
The training of the model is done by splitting the dataset for train and testing with ratio of 0.2 (160 for training and 40 for testing). The 160 data for training then augmented by rotating it 15 degree and 345 degree, thus tripling the data into 480. All the features (`symmetry`, `std_color`, and `1/circularity`) in the train and test is calculated. Then the model is trained using the train data (including the augmented data).

#### **Model Pipeline**
the process to "predict" the skin lesion image with the model
1. Image acquisition
2. Skin lesions segmentation
3. Calculate value of `symmetry`, `std_color`, and `1/circularity` using the original and segmented image as a mask
4. Treat the values as feature vectors then predict with the model

