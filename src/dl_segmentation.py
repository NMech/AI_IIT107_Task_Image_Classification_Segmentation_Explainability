# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import sys
import cv2

#%%
root_directory = os.getcwd() # Run from current directory
sys.path.append("./import_pys")
sys.path.append("./import_pys/Plots")

from ClassificationMetricsPlot   import ClassificationMetricsPlot
from image_manipulator           import read_images, resize_images, read_mask_images
from tabulate                    import tabulate

import numpy               as np
import tensorflow          as tf
import matplotlib.pyplot   as plt
import matplotlib.cm       as cm
import pandas              as pd

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from sklearn.metrics                       import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection               import train_test_split
from tensorflow.keras.models               import Model
from tensorflow.keras.applications.vgg19   import VGG19
from tensorflow.keras.applications.vgg19   import preprocess_input
from tensorflow.keras.utils                import array_to_img, img_to_array
from IPython.display                       import Image, display


#%%
###############################################################################
################################## Input data #################################
###############################################################################
dataset_normal_path    = rf"{root_directory}/../dataset/initial/normal/"
dataset_benign_path    = rf"{root_directory}/../dataset/initial/benign/"
dataset_malignant_path = rf"{root_directory}/../dataset/initial/malignant/"

fig_filepath           = rf"{root_directory}/../Report/Figures/dl_segmentation"
model_filepath         = rf"{root_directory}/../models/dl_segmentation"
res_filepath           = rf"{root_directory}/../results/dl_segmentation"
plotDiagrams           = True
saveDiagrams           = True

# Set the image and model parameters
image_size = (224, 224)
num_classes = 3
train_ratio = 0.8  # 80% for training, 20% for testing

Train = False

np.random.seed(42)
#%%
print("Images reading Began") 
normal_images    = read_images(dataset_normal_path)
benign_images    = read_images(dataset_benign_path)
malignant_images = read_images(dataset_malignant_path)
print("Images reading Done")
#%%
print("Mask Images reading Began") 
normal_images_masks    = read_mask_images(dataset_normal_path)
benign_images_masks    = read_mask_images(dataset_benign_path)
malignant_images_masks = read_mask_images(dataset_malignant_path)
print("Mask Images reading Done")

#%%
print("Images Resizing Began")
normal_images    = resize_images(normal_images, image_size)
benign_images    = resize_images(benign_images, image_size)
malignant_images = resize_images(malignant_images, image_size)
print("Images Resizing Done")
#%%
print("Mask Images Resizing Began")
normal_images_masks    = resize_images(normal_images_masks, image_size)
benign_images_masks    = resize_images(benign_images_masks, image_size)
malignant_images_masks = resize_images(malignant_images_masks, image_size)
print("Mask Images Resizing Done")
#%%
print("Mask Images From 255 to 1 Began") 
for idx in range(len(normal_images_masks)):    normal_images_masks[idx]    = np.where(normal_images_masks[idx] > 0, 1, normal_images_masks[idx])
for idx in range(len(benign_images_masks)):    benign_images_masks[idx]    = np.where(benign_images_masks[idx] > 0, 1, benign_images_masks[idx])
for idx in range(len(malignant_images_masks)): malignant_images_masks[idx] = np.where(malignant_images_masks[idx] > 0, 1, malignant_images_masks[idx])
print("Mask Images From 255 to 1 Done") 
#%%
#baseline_accuracy
zeros = 0
ones  = 0
for elem in normal_images_masks:
    for row in elem:
        for pixel in row:
            if pixel == 0 : zeros +=1
            else: ones +=1
for elem in benign_images_masks:
    for row in elem:
        for pixel in row:
            if pixel == 0 : zeros +=1
            else: ones +=1
for elem in malignant_images_masks:
    for row in elem:
        for pixel in row:
            if pixel == 0 : zeros +=1
            else: ones +=1
baseline_accuracy = max(zeros,ones) / float(ones+zeros)
with open(rf"{res_filepath}/baseline_accuracy.dat","w") as fileOut:
    str = f"baseline_accuracy: {baseline_accuracy}"
    fileOut.write(str)
#%%
print("Convert Images to RGB Began")
for idx in range(len(normal_images)):    normal_images[idx]    = cv2.cvtColor(normal_images[idx], cv2.COLOR_GRAY2RGB)
for idx in range(len(benign_images)):    benign_images[idx]    = cv2.cvtColor(benign_images[idx], cv2.COLOR_GRAY2RGB)
for idx in range(len(malignant_images)): malignant_images[idx] = cv2.cvtColor(malignant_images[idx], cv2.COLOR_GRAY2RGB)
print("Convert Images to RGB Done")

print("Preprocess Images Began")
for idx in range(len(normal_images)):    normal_images[idx]    = preprocess_input(normal_images[idx])
for idx in range(len(benign_images)):    benign_images[idx]    = preprocess_input(benign_images[idx])
for idx in range(len(malignant_images)): malignant_images[idx] = preprocess_input(malignant_images[idx])
print("Preprocess Images Done")

print("Split to train-test and tranform Began")
#%%
train_normal_images,    test_normal_images,    train_normal_images_masks,    test_normal_images_masks    \
                                 = train_test_split(normal_images, normal_images_masks, train_size=train_ratio, test_size=1-train_ratio, random_state=42)
train_benign_images,    test_benign_images,    train_benign_images_masks,    test_benign_images_masks    \
                                 = train_test_split(benign_images, benign_images_masks, train_size=train_ratio, test_size=1-train_ratio, random_state=42)
train_malignant_images, test_malignant_images, train_malignant_images_masks, test_malignant_images_masks \
                                 = train_test_split(malignant_images, malignant_images_masks, train_size=train_ratio, test_size=1-train_ratio, random_state=42)

#%%
train_dataset_images = train_normal_images + train_benign_images + train_malignant_images
test_dataset_images  = test_normal_images  + test_benign_images  + test_malignant_images

train_dataset_images_masks = train_normal_images_masks + train_benign_images_masks + train_malignant_images_masks
test_dataset_images_masks  = test_normal_images_masks  + test_benign_images_masks  + test_malignant_images_masks

#%%
# Load the pre-trained VGG19 model
print("Load the pre-trained VGG19 Began")
vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("Load the pre-trained VGG19 Done")

# Freeze the base model's layers
print("Freeze the base model's layers Began")
for layer in vgg19_model.layers:
    layer.trainable = False
print("Freeze the base model's layers Done")


# Get the last convolutional layer from the vgg19_model
last_layer = vgg19_model.get_layer('block5_conv4').output

# Build the decoder part of the U-Net
x = Conv2D(512, 3, activation='relu', padding='same')(last_layer)
x = Conv2D(512, 3, activation='relu', padding='same')(x)
x = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(x)
x = concatenate([vgg19_model.get_layer('block4_conv4').output, x], axis=3)
x = Conv2D(256, 3, activation='relu', padding='same')(x)
x = Conv2D(256, 3, activation='relu', padding='same')(x)
x = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(x)
x = concatenate([vgg19_model.get_layer('block3_conv4').output, x], axis=3)
x = Conv2D(128, 3, activation='relu', padding='same')(x)
x = Conv2D(128, 3, activation='relu', padding='same')(x)
x = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(x)
x = concatenate([vgg19_model.get_layer('block2_conv2').output, x], axis=3)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = Conv2DTranspose(1, 2, strides=(2, 2), padding='same')(x)  # Upsample by a factor of 2
x = Conv2D(1, 1, activation='sigmoid')(x)  # Output with single channel

# Create the U-Net model
model = Model(inputs=vgg19_model.input, outputs=x)

# Compile the model
print("Compile Model Began")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],)
print("Compile Model Done")

train_dataset_images_masks = np.array(train_dataset_images_masks)
train_dataset_images_masks = np.expand_dims(train_dataset_images_masks, axis=-1)
train_dataset_images       = np.array(train_dataset_images)

# Train the model
if Train:
    print("Train Model Began")
    model.fit(train_dataset_images, train_dataset_images_masks, epochs=15, batch_size=32)
    print("Train Model Done")

    #%%
    # Save the model
    model.save(model_filepath + "/dl_segmentation.h5")

#%%
# Load the model
model = tf.keras.models.load_model(model_filepath + "/dl_segmentation.h5")

#%%
# Generate predictions on the test set
test_predictions = model.predict(np.array(test_dataset_images))

#%%
for idx in range(len(test_predictions)):    test_predictions[idx]    = np.where(test_predictions[idx] >= 0.5, 1, test_predictions[idx])
for idx in range(len(test_predictions)):    test_predictions[idx]    = np.where(test_predictions[idx] < 0.5, 0, test_predictions[idx])

#%%
# Calculate Metrics
test_predictions_1D = np.ravel(test_predictions)
test_predictions_1D = np.asarray(test_predictions_1D, dtype = 'uint8')
test_masks_1D       = np.ravel(np.array(test_dataset_images_masks))

accuracy_score_  = accuracy_score(test_masks_1D,  test_predictions_1D)
precision_score_ = precision_score(test_masks_1D, test_predictions_1D, average=None)
recall_score_    = recall_score(test_masks_1D, test_predictions_1D, average=None)
f1_score_        = f1_score(test_masks_1D, test_predictions_1D, average=None)

print("Accuracy:", accuracy_score_)
print("Precision:", precision_score_)
print("Recall:", recall_score_)
print("F1 Score:", f1_score_)

Test_Set_Results = {"Accuracy" : accuracy_score_, 
                    "Precision": precision_score_, 
                    "Recall"   : recall_score_,
                    "F1_score" : f1_score_}
    
Test_Set_Results_tab = tabulate(pd.DataFrame(Test_Set_Results).round(3).T , headers="keys", tablefmt='grid')  
with open(rf"{res_filepath}/Test_Set_Results.dat","w") as fileOut:
    fileOut.write(Test_Set_Results_tab)
    
#%%
###############################################################################
############################### Confusion Matrix ##############################
###############################################################################    
if plotDiagrams == True:
    clf_metricsPlot = ClassificationMetricsPlot(test_masks_1D)
    CMatTest = confusion_matrix(test_masks_1D, test_predictions_1D)
    clf_metricsPlot.Confusion_Matrix_Plot(test_predictions_1D, CMatTest, normalize=True,
                                          labels=["normal","benign-malignant"],
                                          cMap="default",Rotations=[0.,0.],
                                          savePlot=[saveDiagrams,fig_filepath,"Confusion_Matrix"]) 

#%%
# Choose a random sample from the test set
sample_index = np.random.randint(0, len(test_dataset_images))


# Retrieve the original image and its corresponding predicted mask
original_image = test_dataset_images[sample_index]
original_image_mask = test_dataset_images_masks[sample_index]
predicted_mask = test_predictions[sample_index]


# Display the original image
plt.figure(figsize=(8, 8))
plt.subplot(1, 4, 1)
plt.imshow(original_image_mask, cmap='gray')
plt.title('Orig_mask')

# Display the predicted mask
plt.subplot(1, 4, 2)
plt.imshow(predicted_mask[:, :, 0], cmap='gray')
plt.title('Pred_mask')

# Display the predicted mask
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.title('orig')

# Explainability
sample_index = np.random.randint(0, len(test_dataset_images))
def get_grad_cam(image, model):
    image = np.expand_dims(image, axis=0)
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        [model.inputs], [model.get_layer("block5_conv4").output, model.output]
    )
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image)
        pred_index = tf.argmax(preds[0]).numpy()[0][0]
        class_channel = preds[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Generate the Grad-CAM heatmap
cam = get_grad_cam(original_image, model)

# Rescale heatmap to a range 0-255
heatmap = np.uint8(255 * cam)
jet = cm.get_cmap("jet")
 # Use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
# Create an image with RGB colorized heatmap
jet_heatmap = array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((original_image.shape[1], original_image.shape[0]))
jet_heatmap = img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + original_image
superimposed_img = array_to_img(superimposed_img)

# Display Grad CAM
# Display the Grad CAM
plt.subplot(1, 4, 4)
plt.imshow(superimposed_img)
plt.title('explain')
plt.show()
