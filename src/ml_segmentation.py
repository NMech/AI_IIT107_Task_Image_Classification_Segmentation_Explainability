# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action="ignore")

import os
import sys
import joblib
from   tabulate import tabulate
#%%
root_directory = os.getcwd() # Run from current directory
sys.path.append("./import_pys")
sys.path.append("./import_pys/Plots")

from data_preprocessing            import Pipeline_preprocessor
from FeatureExtractor              import extract_features_per_pixel
from image_manipulator             import read_images, read_mask_images, resize_images, crop_around_tumor
from ClassificationMetricsPlot     import ClassificationMetricsPlot
from main_funcs                    import Classifier_Fit, Save_Classifier
from FeaturesHistogramPlot         import FeaturesHistogramPlot
from sklearn.decomposition         import PCA

#%%
import pandas              as pd
import numpy               as np
import matplotlib.pyplot   as plt
import time

#%%
from sklearn.model_selection       import train_test_split
from sklearn.metrics               import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm                   import SVC, LinearSVC
from sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing         import PolynomialFeatures
from sklearn.neighbors             import KNeighborsClassifier

#%%
np.random.seed(42)
#%%
###############################################################################
################################## Input data #################################
###############################################################################
dataset_benign_path    = rf"{root_directory}/../dataset/ml_segmentation/benign/"
dataset_malignant_path = rf"{root_directory}/../dataset/ml_segmentation/malignant/"
fig_filepath           = rf"{root_directory}/../Report/Figures/ml_segmentation"
model_filepath         = rf"{root_directory}/../models/ml_segmentation"
res_filepath           = rf"{root_directory}/../results/ml_segmentation"
plotDiagrams      = True
saveDiagrams      = True
RunModelSelection = True # True only for choosing the final model hyperparameters (Time consuming)

testSize                   = 0.2
Cv                         = 5 # cross-validation folds
dimension                  = 128
image_dimensions           = (dimension, dimension)
#%%
###############################################################################
######################### Creation of plotting objects ########################
###############################################################################
histPlotObj    = FeaturesHistogramPlot()
#%%
###############################################################################
############################ Read data & get a look ###########################
###############################################################################
print("Images reading Began") 
benign_images    = read_images(dataset_benign_path)
malignant_images = read_images(dataset_malignant_path)
print("Images reading Done")

print("Mask Images reading Began")
benign_images_masks    = read_mask_images(dataset_benign_path)
malignant_images_masks = read_mask_images(dataset_malignant_path)
print("Mask Images reading Done")

print("Crop Images Began")
for idx in range(len(benign_images)):
    benign_images[idx], benign_images_masks[idx] = crop_around_tumor(benign_images[idx], benign_images_masks[idx])
    
for idx in range(len(malignant_images)):
    malignant_images[idx], malignant_images_masks[idx] = crop_around_tumor(malignant_images[idx], malignant_images_masks[idx])
print("Crop Images Done")
#%%
total_num_of_images = len(malignant_images) + len(benign_images) 
print("Number of Images = ", total_num_of_images)
#%%
print("Images Resizing Began")
benign_images    = resize_images(benign_images, image_dimensions)
malignant_images = resize_images(malignant_images, image_dimensions)
print("Images Resizing Done")

print("Mask Images Resizing Began")
benign_images_masks    = resize_images(benign_images_masks, image_dimensions)
malignant_images_masks = resize_images(malignant_images_masks, image_dimensions)
print("Mask Images Resizing Done")
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
print("Mask Images flatten Began")
#%%
num_of_pixels       = 0
num_of_tumor_pixels = 0
for image in benign_images_masks_flat:
    for pixel in image:
        num_of_pixels += 1
        if pixel != 0: num_of_tumor_pixels += 1
for image in malignant_images_masks_flat:
    for pixel in image:
        num_of_pixels += 1
        if pixel != 0: num_of_tumor_pixels += 1
#%%
###############################################################################
############################### Baseline Accuracy #############################
###############################################################################
with open(rf"{res_filepath}/baseline.dat","w") as fileOut:
    fileOut.write(f"baseline_accuracy = {round(((num_of_pixels-num_of_tumor_pixels)/num_of_pixels)*100, 2)}%")
#%%
###############################################################################
############################### Extract  Features #############################
###############################################################################
benign_images_features    = []
malignant_images_features = []               
print("Features Extraction Began")
perc_idx = 0
# produces a list of features for each pixel
start = time.time()
for idx in range(len(benign_images)):    
    benign_images_features.append(extract_features_per_pixel(benign_images[idx]))
    perc_idx += 1
    if perc_idx % 5 == 0: 
        end = time.time()
        perc = round((float(perc_idx)/total_num_of_images)*100, 2)
        print(f"progress = {perc}% - time {round(end-start,2)}sec")
        start = end
for idx in range(len(malignant_images)): 
    malignant_images_features.append(extract_features_per_pixel(malignant_images[idx]))
    perc_idx += 1
    if perc_idx % 5 == 0: 
        end = time.time()
        perc = round((float(perc_idx)/total_num_of_images)*100, 2)
        print(f"progress = {perc}% - time {round(end-start,2)}sec")
        start = end
print("progress = 100%")
print("Features Extraction Done")
#%% 
num_of_features = len(benign_images_features[0][0])
print("Number of features = ", num_of_features)
print("Number of features = ", num_of_features)
with open(rf"{res_filepath}/num_of_features.dat","w") as fileOut:
    str = f"Num_of_features: {num_of_features}"
    fileOut.write(str)
#%%
###############################################################################
##################### Create Images dataframes with labels ####################
###############################################################################
print("Create Images dataframes with labels Began")
tmp_list = []
ColumnsUsed = [f"feature_{i+1}" for i in range(num_of_features)]
data_df = pd.DataFrame(columns=ColumnsUsed + ["label"] )
for i in range(len(benign_images_features)):           # image
    for j in range(len(benign_images_masks_flat[i])):  # pixel
        label = 0
        if benign_images_masks_flat[i][j] != 0: 
            label = 1
        ll = []
        for elem in benign_images_features[i][j]: ll.append(elem) 
        ll.append(label)
        tmp_list.append(ll)
for i in range(len(malignant_images_features)):           # image
    for j in range(len(malignant_images_masks_flat[i])):  # pixel
        label = 0
        if malignant_images_masks_flat[i][j] != 0: 
            label = 1
        ll = []
        for elem in malignant_images_features[i][j]: ll.append(elem) 
        ll.append(label)
        tmp_list.append(ll)
data_df = pd.DataFrame(tmp_list, columns=ColumnsUsed + ["label"])
print("Create Images dataframes with labels Done")
#%%
###############################################################################
########################### Split dataset train-test ##########################
###############################################################################
X        = pd.DataFrame(data_df[ColumnsUsed], columns=ColumnsUsed)
y        = data_df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, shuffle=True, random_state=42)  
preprocess_pipeline = Pipeline_preprocessor([], ColumnsUsed)   
joblib.dump(preprocess_pipeline, open(f"{model_filepath}\preprocess_pipeline.joblib", "wb")) #save fitted and transformed pipeline

#%%
###############################################################################
####################### Data cleaning & transformations #######################
###############################################################################   
print("Data cleaning & transformations Began")                                          
X_train  = pd.DataFrame(X_train, columns=ColumnsUsed)
y_train  = pd.Series(y_train, name="label")
y_train.index = [i for i in range(len(y_train))]
X_train  = pd.DataFrame(preprocess_pipeline.fit_transform(X_train), columns=ColumnsUsed)     # X_train_prepared from now on
joblib.dump(preprocess_pipeline, open(f"{model_filepath}\preprocess_pipeline.joblib", "wb")) #save fitted and transformed pipeline
XYconcat_train = pd.concat([X_train, y_train], axis=1)                                       # DataFrame used in training procedure
if plotDiagrams == True and num_of_features < 20:
    histPlotObj.HistPlot(XYconcat_train, "label", savePlot=[saveDiagrams, fig_filepath, "Features_Histogram"])  
print("Data cleaning & transformations Done")                                          
#%%
#-----------------------------------------------------------------------------#
###############################################################################
######################## Testing different classifiers ########################
###############################################################################
models_CV_res = []  
#%%
###############################################################################
################################## Linear SVM #################################
###############################################################################
svm_clf = LinearSVC(C=1, loss="hinge")
clf_txt = "1 Linear SVM"
svm_clf, score = Classifier_Fit(svm_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
################################ Polynomial SVM ###############################
###############################################################################
pol  = PolynomialFeatures(degree=2)
Xpol = pol.fit_transform(X_train)
pol_svm_clf = LinearSVC(C=1, loss="hinge")
clf_txt     = "2 Polynomial SVM"
pol_svm_clf, score = Classifier_Fit(pol_svm_clf, Xpol, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
################################ Random Forest ################################
###############################################################################
rnd_clf = RandomForestClassifier(n_estimators=200, max_features="sqrt", max_depth=10, max_leaf_nodes=None, n_jobs=-1)
clf_txt = "3 Random Forest"
rnd_clf, score = Classifier_Fit(rnd_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
feature_importance = rnd_clf.feature_importances_
#%%
###############################################################################
################################## Ada Boost ################################## 
###############################################################################
ada_clf = AdaBoostClassifier()
clf_txt = "4 Ada Boost"
ada_clf, score = Classifier_Fit(ada_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
############################ k-Neighbors classifier ###########################
###############################################################################
kNeighbor_clf  = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
clf_txt = "5 k-Neighbors"
kNeighbor_clf, score = Classifier_Fit(kNeighbor_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
models_CV_res_tab = tabulate(models_CV_res, tablefmt='grid')
with open(rf"{res_filepath}/models_CV_res.dat","w") as fileOut:
    fileOut.write(models_CV_res_tab)
#%%
###############################################################################
############################## Create final model #############################
###############################################################################
final_model_clf = VotingClassifier(estimators=[
                                     ("pol_svm",  pol_svm_clf),
                                     ("lin_svm",  svm_clf),
                                     ("rnd_clf",  rnd_clf),
                                    ], voting="hard")  
final_model_clf.fit(X_train, y_train) 

Save_Classifier(pol_svm_clf,     model_filepath, "Polynomial SVM")
Save_Classifier(svm_clf,         model_filepath, "Linear SVM")
Save_Classifier(rnd_clf,         model_filepath, "Random Forest")
Save_Classifier(final_model_clf, model_filepath, "Voting_Classifier") 
#%%
###############################################################################
################################ Test Set scores ##############################
###############################################################################
Test_Set_Results = {}
X_test_prepared = pd.DataFrame(preprocess_pipeline.transform(X_test), columns=ColumnsUsed)                                                                              
for clf_name, clf in [("pol_svm", pol_svm_clf), ("lin_svm", svm_clf), ("rnd_clf", rnd_clf), ("final_model", final_model_clf)]:
    if clf_name == "pol_svm":
        y_Pred_test      = clf.predict(pol.fit_transform(X_test_prepared))
    else:
        y_Pred_test      = clf.predict(X_test_prepared)
    accuracy_score_  = accuracy_score(y_test,  y_Pred_test)
    precision_score_ = precision_score(y_test, y_Pred_test, average=None)
    recall_score_    = recall_score(y_test, y_Pred_test, average=None)
    f1_score_        = f1_score(y_test, y_Pred_test, average=None)

    Test_Set_Results[clf_name] = {"Accuracy" : accuracy_score_, 
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
    clf_metricsPlot = ClassificationMetricsPlot(y_test)
    CMatTest = confusion_matrix(y_test, y_Pred_test)
    clf_metricsPlot.Confusion_Matrix_Plot(y_Pred_test, CMatTest, normalize=True,
                                          labels=["normal","benign-malignant"],
                                          cMap="default",Rotations=[0.,0.],
                                          savePlot=[saveDiagrams,fig_filepath,"Confusion_Matrix"]) 
