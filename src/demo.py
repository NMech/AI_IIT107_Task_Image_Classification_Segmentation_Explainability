# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action="ignore")
import joblib
import os
import sys

import numpy              as np
import pandas             as pd
import matplotlib.pyplot  as plt

#%%
root_directory = os.getcwd() # Run from current directory
sys.path.append("./import_pys")
sys.path.append("./import_pys/Plots")

from FeatureExtractor              import extract_features_per_pixel
from image_manipulator             import read_images, read_mask_images, resize_images, crop_around_tumor
from PIL                           import Image

#%%
np.random.seed(42)
#%%
dataset_benign_path    = rf"{root_directory}/../dataset/ml_segmentation/for_demo_benign/"
dataset_malignant_path = rf"{root_directory}/../dataset/ml_segmentation/for_demo_malignant/"

model_filepath      = rf"{root_directory}/../models/ml_segmentation/"
model_filename      = "Voting_Classifier.joblib"
preprocess_filename = "preprocess_pipeline.joblib"
separator           = ","
ColumnsUsed = [f"feature_{i+1}" for i in range(22)]


dimension        = 128
image_dimensions = (dimension,dimension)

print("Images reading Began") 
benign_images    = read_images(dataset_benign_path)
malignant_images = read_images(dataset_malignant_path)
print("Images reading Done")

print("Mask Images reading Began")
benign_images_masks    = read_mask_images(dataset_benign_path)
malignant_images_masks = read_mask_images(dataset_malignant_path)
print("Mask Images reading Done")

#%%
total_num_of_images = len(malignant_images) + len(benign_images) 
print("Number of Images = ", total_num_of_images)

print("Crop Images Began")
for idx in range(len(benign_images)):
    benign_images[idx], benign_images_masks[idx] = crop_around_tumor(benign_images[idx], benign_images_masks[idx])
    
for idx in range(len(malignant_images)):
    malignant_images[idx], malignant_images_masks[idx] = crop_around_tumor(malignant_images[idx], malignant_images_masks[idx])
print("Crop Images Done")

print("Images Resizing Began")
benign_images    = resize_images(benign_images, image_dimensions)
malignant_images = resize_images(malignant_images, image_dimensions)
print("Images Resizing Done")
print("Mask Images Resizing Began")
benign_images_masks    = resize_images(benign_images_masks, image_dimensions)
malignant_images_masks = resize_images(malignant_images_masks, image_dimensions)
print("Mask Images Resizing Done")
#%%
###############################################################################
############################### Extract  Features #############################
###############################################################################                                 
benign_images_features    = []
malignant_images_features = []
print("Features Extraction Began")
# produces a list of features for each pixel
for idx in range(len(benign_images)):     benign_images_features.append(extract_features_per_pixel(benign_images[idx]))
for idx in range(len(malignant_images)):  malignant_images_features.append(extract_features_per_pixel(malignant_images[idx]))
print("Features Extraction Done")
#%% 
benign_images_flat          = []
malignant_images_flat       = []
benign_images_masks_flat    = []
malignant_images_masks_flat = []

print("Images flatten Began")
for idx in range(len(benign_images)):    benign_images_flat.append(benign_images[idx].flatten())
for idx in range(len(malignant_images)): malignant_images_flat.append(malignant_images[idx].flatten())
print("Images flatten Began")

print("Mask Images flatten Began")
for idx in range(len(benign_images_masks )):    benign_images_masks_flat.append(benign_images_masks[idx].flatten())
for idx in range(len(malignant_images_masks)):  malignant_images_masks_flat.append(malignant_images_masks[idx].flatten())
print("Mask Images flatten Done")
#%%
model_fileInp           = os.path.join(model_filepath, model_filename)
pipe_preprocess_fileInp = os.path.join(model_filepath, preprocess_filename)
#%%
model    = joblib.load(model_fileInp)
pipeline = joblib.load(pipe_preprocess_fileInp)
#%%
total_num_of_images = len(malignant_images) \
                    + len(benign_images) 
#%%
print("Assess Images Began")
tmp_list = []
data_df = pd.DataFrame(columns=ColumnsUsed)
for image in benign_images_features:  # image
    for pixel in image:               # pixel
        ll = [pixel[0], pixel[1], pixel[2], pixel[3], pixel[4], pixel[5], pixel[6], pixel[7], pixel[8], pixel[9], pixel[10], \
              pixel[11], pixel[12], pixel[13], pixel[14], pixel[15], pixel[16], pixel[17], pixel[18], pixel[19], pixel[20], pixel[21]]
        tmp_list.append(ll)
for image in malignant_images_features:  # image
    for pixel in image:                  # pixel
        ll = [pixel[0], pixel[1], pixel[2], pixel[3], pixel[4], pixel[5], pixel[6], pixel[7], pixel[8], pixel[9], pixel[10], \
              pixel[11], pixel[12], pixel[13], pixel[14], pixel[15], pixel[16], pixel[17], pixel[18], pixel[19], pixel[20], pixel[21]]
        tmp_list.append(ll)

X_test_demo     = pd.DataFrame(tmp_list, columns=ColumnsUsed)
X_test_prepared = pipeline.transform(X_test_demo)
prediction_demo = model.predict(X_test_prepared)
print("Assess Images Done")
#%%
images = []
images_assessed = np.split(prediction_demo,total_num_of_images)
for image in images_assessed:
    image_res_list = []
    image_list = image.tolist()
    for pixel in image_list:
        if pixel == 0: image_res_list.append(0)
        else: image_res_list.append(255)
        
    arr = np.array(image_res_list, dtype=np.uint8)
    arr = np.reshape(arr, (-1, dimension))
    images.append(arr)
#%%
# Display the original image
plt.figure(figsize=(8, 8))
plt.subplot(1, 3, 1)
plt.imshow(malignant_images[0], cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(malignant_images_masks[0], cmap='gray')
plt.title('Original Image mask')

# Display the predicted mask
plt.subplot(1, 3, 3)
plt.imshow(images[2], cmap='gray')
plt.title('Predicted Mask')

plt.show()



        
