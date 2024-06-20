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