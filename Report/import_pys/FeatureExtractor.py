import mahotas                  as mt
import numpy                    as np


from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import prewitt_h,prewitt_v
from skimage.measure import shannon_entropy
from scipy.stats     import skew, kurtosis, entropy


import cv2

def extract_Haralick_features(image):
    
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return list(ht_mean)

def extract_HOG_features(image):
    
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return list(fd)

def extract_SIFT_features(image):
    
    surf = cv2.SIFT_create(nfeatures = 100)
    keypoints_sift, des = surf.detectAndCompute(image, None)
    return keypoints_sift

def extract_Prewitt_features(image):
    
    edges_prewitt_horizontal = prewitt_h(image)
    edges_prewitt_vertical = prewitt_v(image)
    return np.concatenate((edges_prewitt_horizontal, edges_prewitt_vertical))
   # imshow(edges_prewitt_vertical, cmap='gray')
   
def extract_GLCM_features(image):
    
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast    = graycoprops(glcm, 'contrast')[0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0]
    energy      = graycoprops(glcm, 'energy')[0]
    correlation = graycoprops(glcm, 'correlation')[0]
    
    features = []
    features.append(contrast[0])
    features.append(homogeneity[0])
    features.append(energy[0])
    features.append(correlation[0])
    return features

def extract_GLRLM_features(image):
    
    glrlm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Compute GLRLM features
    glrlm_features = []
    glrlm_features.append(graycoprops(glrlm, 'energy')[0][0])
    glrlm_features.append(graycoprops(glrlm, 'entropy')[0][0])
    glrlm_features.append(graycoprops(glrlm, 'contrast')[0][0])
    glrlm_features.append(graycoprops(glrlm, 'dissimilarity')[0][0])
    
    return glrlm_features

def extract_GLSZM_features(image):
    
    glszm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Compute GLSZM features
    glszm_features = []
    glszm_features.append(graycoprops(glszm, 'energy')[0][0])
    glszm_features.append(graycoprops(glszm, 'correlation')[0][0])
    glszm_features.append(graycoprops(glszm, 'contrast')[0][0])
    glszm_features.append(graycoprops(glszm, 'dissimilarity')[0][0])
    glszm_features.append(graycoprops(glszm, 'homogeneity')[0][0])
    glszm_features.append(graycoprops(glszm, 'ASM')[0][0])
    glszm_features.append(shannon_entropy(glszm.reshape(-1)))
    
    return glszm_features

def extract_LBP_features(image):
    
    radius = 3  # LBP radius
    n_points = 8 * radius  # number of LBP points
    lbp = local_binary_pattern(image, n_points, radius)

    # Compute histogram of LBP features
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return list(hist)

def extract_features_per_pixel(image):
    
    # Initialize an empty feature matrix
    features = np.zeros((image.shape[0], image.shape[1], 21))

    # Loop through every pixel in the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            
            # Create a 5x5 patch around the current pixel
            patch = image[max(0, row-2):min(row+3, image.shape[0]), max(0, col-2):min(col+3, image.shape[1])]
            
            textures = mt.features.haralick(patch)
            ht_mean = list(textures.mean(axis=0))
            features[row, col, 0] = ht_mean[0]
            features[row, col, 1] = ht_mean[1]
            features[row, col, 2] = ht_mean[2]
            features[row, col, 3] = ht_mean[3]
            features[row, col, 4] = ht_mean[4]
            features[row, col, 5] = ht_mean[5]
            features[row, col, 6] = ht_mean[6]
            features[row, col, 7] = ht_mean[7]
            features[row, col, 8] = ht_mean[8]
            features[row, col, 9] = ht_mean[9]
            features[row, col, 10] = ht_mean[10]
            features[row, col, 11] = ht_mean[11]
            features[row, col, 12] = ht_mean[12]
            
            # Calculate the intensity features for the patch
            intensity_values = patch.flatten()
            features[row, col, 13] = np.mean(intensity_values)
            features[row, col, 14] = np.std(intensity_values)
            features[row, col, 15] = np.median(intensity_values)
            features[row, col, 16] = np.max(intensity_values) - np.min(intensity_values)
            features[row, col, 17] = skew(intensity_values)
            features[row, col, 18] = kurtosis(intensity_values)
            features[row, col, 19] = np.sum(np.square(intensity_values))
            features[row, col, 20] = entropy(intensity_values)
            
    features = features.reshape((-1, 21)).tolist()
    
    # Create LBP features
    radius = 2  
    n_points = 8 * radius  
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp = lbp.flatten()
    
    # combine features
    for idx in range(len(features)): features[idx].append(lbp[idx])
    
    return features

