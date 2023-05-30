import os
import cv2
import numpy               as np
import matplotlib.pyplot   as plt

def read_images(folder_path):
    images = []
    for image_str in os.listdir(folder_path):
        if image_str.endswith(".png") and not "_mask" in image_str:
            image = cv2.imread(folder_path+image_str, cv2.IMREAD_GRAYSCALE)
            images.append(image)
    return images

def read_mask_images(folder_path):
    images = []
    last_string = ""
    for image_str in os.listdir(folder_path):
        if image_str.endswith(".png") and "_mask" in image_str:
            first_part_str = image_str.split("_mask")[0]
            image = cv2.imread(folder_path+image_str, cv2.IMREAD_GRAYSCALE)
            if len(images) > 0:
                first_part_str_last = last_string.split("_mask")[0]
                if first_part_str == first_part_str_last:
                    images[-1] = images[-1] + image
                else:
                    images.append(image)
                    last_string = image_str 
            else:
                images.append(image)
                last_string = image_str
    return images

def resize_images(images, size_tuple):
    images_list = []
    for idx in range(len(images)):    
        images_list.append(cv2.resize(images[idx], size_tuple))
    return images_list

def crop_around_tumor(image, mask):
    
    # Find the bounding box of the tumor region in the ultrasound image
    rows, cols = np.nonzero(mask)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Add a padding margin to the bounding box
    margin = 10
    min_row = max(0, min_row - margin)
    max_row = min(image.shape[0], max_row + margin)
    min_col = max(0, min_col - margin)
    max_col = min(image.shape[1], max_col + margin)
    
    # Crop the ultrasound image and the mask
    cropped_image = image[min_row:max_row, min_col:max_col]
    cropped_mask = mask[min_row:max_row, min_col:max_col]
    return cropped_image, cropped_mask
    
# =============================================================================
#     # Display the cropped images
#     cv2.imshow("Cropped Image", cropped_image)
#     cv2.imshow("Cropped Mask", cropped_mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# =============================================================================
