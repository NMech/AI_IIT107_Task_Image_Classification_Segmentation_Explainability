# -*- coding: utf-8 -*-
import os
import sys
import cv2

#%%
root_directory = os.getcwd() # Run from current directory
sys.path.append("./import_pys")
sys.path.append("./import_pys/Plots")

from image_manipulator  import read_images, resize_images
from tabulate           import tabulate
from lime               import lime_image

import numpy        as np
import tensorflow   as tf
import pandas       as pd
import matplotlib.pyplot   as plt

from ClassificationMetricsPlot             import ClassificationMetricsPlot
from sklearn.metrics                       import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection               import train_test_split
from skimage.segmentation                  import mark_boundaries
from tensorflow.keras.applications.vgg19   import VGG19
from tensorflow.keras.applications.vgg19   import preprocess_input
from tensorflow.keras                      import layers, models
from tensorflow.keras.utils                import to_categorical
from tensorflow.keras.callbacks            import EarlyStopping
from tensorflow.keras.preprocessing.image  import ImageDataGenerator

#%%
###############################################################################
################################## Input data #################################
###############################################################################
dataset_normal_path    = rf"{root_directory}/../dataset/initial/normal/"
dataset_benign_path    = rf"{root_directory}/../dataset/initial/benign/"
dataset_malignant_path = rf"{root_directory}/../dataset/initial/malignant/"

fig_filepath           = rf"{root_directory}/../Report/Figures/dl_classification"
model_filepath         = rf"{root_directory}/../models/dl_classification"
res_filepath           = rf"{root_directory}/../results/dl_classification"
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
#baseline_accuracy
baseline_accuracy = max(len(normal_images),len( benign_images), len(malignant_images)) / float(len(normal_images)+len( benign_images)+len(malignant_images))
with open(rf"{res_filepath}/baseline_accuracy.dat","w") as fileOut:
    str = f"baseline_accuracy: {baseline_accuracy}"
    fileOut.write(str)
#%%
print("Images Resizing Began")
normal_images    = resize_images(normal_images, image_size)
benign_images    = resize_images(benign_images, image_size)
malignant_images = resize_images(malignant_images, image_size)
print("Images Resizing Done")

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
train_normal_images,    test_normal_images    = train_test_split(normal_images,    train_size=train_ratio, test_size=1-train_ratio, random_state=42)
train_benign_images,    test_benign_images    = train_test_split(benign_images,    train_size=train_ratio, test_size=1-train_ratio, random_state=42)
train_malignant_images, test_malignant_images = train_test_split(malignant_images, train_size=train_ratio, test_size=1-train_ratio, random_state=42)

#%%
init_len_normal_train    = len(train_normal_images)
init_len_benign_train    = len(train_benign_images)
init_len_malignant_train = len(train_malignant_images)

print("Normal Train Images Before Augmentation: ", init_len_normal_train)
print("Benign Train Images Before Augmentation: ", init_len_benign_train)
print("Malignant Train Images Before Augmentation: ", init_len_malignant_train)

print("Augment Training Data Began")
datagen = ImageDataGenerator(rotation_range=45, fill_mode='nearest')

for i in range(init_len_normal_train):
    augmented_images = datagen.flow(x=np.expand_dims(train_normal_images[i], axis=0), y=None, batch_size=1)
    for i in range(5): train_normal_images.append(np.squeeze(augmented_images.next()))
    
for i in range(init_len_normal_train):
    augmented_images = datagen.flow(x=np.expand_dims(train_benign_images[i], axis=0), y=None, batch_size=1)
    for i in range(5): train_benign_images.append(np.squeeze(augmented_images.next()))
    
for i in range(init_len_normal_train):
    augmented_images = datagen.flow(x=np.expand_dims(train_malignant_images[i], axis=0), y=None, batch_size=1)
    for i in range(5): train_malignant_images.append(np.squeeze(augmented_images.next()))

print("Normal Train Images After Augmentation: ", len(train_normal_images))
print("Benign Train Images After Augmentation: ", len(train_benign_images))
print("Malignant Train Images After Augmentation: ", len(train_malignant_images))

print("Augment Training Data Done")

#%%
train_normal_labels    = [0] * len(train_normal_images)
train_benign_labels    = [1] * len(train_benign_images)
train_malignant_labels = [2] * len(train_malignant_images)

test_normal_labels    = [0] * len(test_normal_images)
test_benign_labels    = [1] * len(test_benign_images)
test_malignant_labels = [2] * len(test_malignant_images)

#%%
train_dataset_images = train_normal_images + train_benign_images + train_malignant_images
train_dataset_labels = train_normal_labels + train_benign_labels + train_malignant_labels

test_dataset_images = test_normal_images + test_benign_images + test_malignant_images
test_dataset_labels = test_normal_labels + test_benign_labels + test_malignant_labels

## Transforming labels to correct format
train_dataset_labels = to_categorical(train_dataset_labels, num_classes=num_classes)
print("Split to train-test and tranform Done")

#%%
# Load the pre-trained VGG19 model
print("Load the pre-trained VGG19 Began")
vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("Load the pre-trained VGG19 Done")

# Freeze the base model's layers
print("Freeze the base model's layers Began")
vgg19_model.trainable = False
print("Freeze the base model's layers Done")

# Unfreeze the last two layers
for layer in vgg19_model.layers[-2:]:
    layer.trainable = True

# Add last layers
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(num_classes, activation='softmax')
model = models.Sequential([
    vgg19_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
if Train:
    print("Train Model Began")
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
    model.fit(np.array(train_dataset_images), np.array(train_dataset_labels), epochs=15, validation_split=0.1, batch_size=8, callbacks=[es])
    print("Train Model Done")

    #%%
    # Save the model
    model.save(model_filepath + "/dl_classification.h5")

#%%
# Load the model
model = tf.keras.models.load_model(model_filepath + "/dl_classification.h5")

#%%
image_batch = tf.stack(test_dataset_images)

predictions = model.predict(image_batch)
#%%
val_preds = np.argmax(predictions, axis=1)
#%%
accuracy_score_  = accuracy_score(test_dataset_labels,  val_preds)
precision_score_ = precision_score(test_dataset_labels, val_preds, average=None)
recall_score_    = recall_score(test_dataset_labels, val_preds, average=None)
f1_score_        = f1_score(test_dataset_labels, val_preds, average=None)

#%%
print("Accuracy:",  accuracy_score_)
print("Precision:", precision_score_)
print("Recall:",    recall_score_)
print("F1 Score:",  f1_score_)

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
    clf_metricsPlot = ClassificationMetricsPlot(test_dataset_labels)
    CMatTest = confusion_matrix(test_dataset_labels, val_preds)
    clf_metricsPlot.Confusion_Matrix_Plot(val_preds, CMatTest, normalize=True,
                                          labels=["normal","benign","malignant"],
                                          cMap="default",Rotations=[0.,0.],
                                          savePlot=[saveDiagrams,fig_filepath,"Confusion_Matrix"]) 
    
    
#%%
# explainability

def predict_function(image):
    new_image = np.expand_dims(image, axis=0)
    predictions = model.predict(new_image)
    predictions = predictions.reshape((1, -1))
    return predictions

sample_index = np.random.randint(0, len(test_dataset_images))
image = test_dataset_images[sample_index]
image = image.astype('double')

explainer = lime_image.LimeImageExplainer()

# Explain the image using LIME
explanation = explainer.explain_instance(image = image, classifier_fn = predict_function, top_labels=2, hide_color=0, num_samples=1)

# Get the explanation for the top predicted label
top_label = explanation.top_labels[0]
explanation_img, mask = explanation.get_image_and_mask(top_label, positive_only=False, num_features=5, hide_rest=False)

explanation_img = explanation_img.astype(np.float32)
explanation_img = cv2.cvtColor(explanation_img, cv2.COLOR_BGR2GRAY)
explanation_img = explanation_img.astype(np.double)
# Overlay the mask on the original image
image_with_mask = mark_boundaries(explanation_img / 255.0, mask, color = (1,0,0))

image = image.astype(np.float32)

# Plot the original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Plot the mask
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.axis("off")

# Plot the image with the overlayed mask
plt.subplot(1, 3, 3)
plt.imshow(image_with_mask, cmap='gray')
plt.title("Image with Mask")
plt.axis("off")

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()