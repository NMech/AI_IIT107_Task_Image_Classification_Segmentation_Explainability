# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action="ignore")

import os
import sys
import json
from   tabulate import tabulate
#%%
root_directory = os.getcwd() # Run from current directory
sys.path.append("./import_pys")
sys.path.append("./import_pys/Plots")

from data_preprocessing            import Pipeline_preprocessor
from FeatureExtractor              import extract_Haralick_features, extract_HOG_features, \
                                          extract_Prewitt_features, extract_GLCM_features, \
                                          extract_GLRLM_features, extract_GLSZM_features,  \
                                          extract_LBP_features
from image_manipulator             import read_images, resize_images
from ClassificationMetricsPlot     import ClassificationMetricsPlot
from main_funcs                    import Classifier_Fit
from select_model_parameters       import model_parameters
from FeaturesHistogramPlot         import FeaturesHistogramPlot
from sklearn.decomposition         import PCA

#%%
import pandas              as pd
import numpy               as np
import matplotlib.pyplot   as plt
import time

#%%
from sklearn.model_selection       import train_test_split, GridSearchCV
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
dataset_normal_path    = rf"{root_directory}/../dataset/initial/normal/"
dataset_benign_path    = rf"{root_directory}/../dataset/initial/benign/"
dataset_malignant_path = rf"{root_directory}/../dataset/initial/malignant/"
fig_filepath           = rf"{root_directory}/../Report/Figures/ml_classification"
model_filepath         = rf"{root_directory}/../models/ml_classification"
res_filepath           = rf"{root_directory}/../results/ml_classification"
plotDiagrams      = True
saveDiagrams      = False
RunModelSelection = False # True only for choosing the final model hyperparameters (Time consuming)

HOG_features       = True
SIFT_features      = False
PREWITT_features   = False
Haralick_features  = True
GLCM_features      = False
GLRLM_features     = False
GLSZM_features     = True
LBP_features       = True

testSize                   = 0.2
Cv                         = 5 # cross-validation folds
dimension                  = 256
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
normal_images    = read_images(dataset_normal_path)
benign_images    = read_images(dataset_benign_path)
malignant_images = read_images(dataset_malignant_path)
print("Images reading Done")

total_num_of_images = len(normal_images) + len(benign_images) + len(malignant_images)

print("Images Resizing Began")
normal_images    = resize_images(normal_images, image_dimensions)
benign_images    = resize_images(benign_images, image_dimensions)
malignant_images = resize_images(malignant_images, image_dimensions)
print("Images Resizing Done")

#%%
###############################################################################
############################### Baseline Accuracy #############################
###############################################################################
with open(rf"{res_filepath}/baseline.dat","w") as fileOut:
    fileOut.write(f"baseline_accuracy = {round((max(len(normal_images), len(benign_images), len(malignant_images))/total_num_of_images)*100, 2)}%")

    
#%%
###############################################################################
############################### Extract  Features #############################
###############################################################################
normal_images_features    = []
benign_images_features    = []
malignant_images_features = []
                                
if HOG_features:
    print("HOG Features Extraction Began")
    for image in normal_images:    normal_images_features.append(extract_HOG_features(image))
    for image in benign_images:    benign_images_features.append(extract_HOG_features(image))
    for image in malignant_images: malignant_images_features.append(extract_HOG_features(image))
    print("HOG Features Extraction Done")
    pca = PCA(n_components=30)
    normal_images_features    = pca.fit_transform(normal_images_features).tolist()
    benign_images_features    = pca.fit_transform(benign_images_features).tolist()
    malignant_images_features = pca.fit_transform(malignant_images_features).tolist()
    
if PREWITT_features:
    print("PREWITT Features Extraction Began")
    for image in normal_images:    normal_images_features.append(extract_Prewitt_features(image))
    for image in benign_images:    benign_images_features.append(extract_Prewitt_features(image))
    for image in malignant_images: malignant_images_features.append(extract_Prewitt_features(image))
    print("PREWITT Features Extraction Done")
    
if Haralick_features:
    print("Haralick Features Extraction Began")
    for idx in range(len(normal_images)):    normal_images_features[idx]    += extract_Haralick_features(normal_images[idx])
    for idx in range(len(benign_images)):    benign_images_features[idx]    += extract_Haralick_features(benign_images[idx])
    for idx in range(len(malignant_images)): malignant_images_features[idx] += extract_Haralick_features(malignant_images[idx])
    print("Haralick Features Extraction Done")

if GLCM_features:
    print("GLCM Features Extraction Began")
    for idx in range(len(normal_images)):    normal_images_features[idx]    += extract_GLCM_features(normal_images[idx])
    for idx in range(len(benign_images)):    benign_images_features[idx]    += extract_GLCM_features(benign_images[idx])
    for idx in range(len(malignant_images)): malignant_images_features[idx] += extract_GLCM_features(malignant_images[idx])
    print("GLCM Features Extraction Done")
    
if GLRLM_features:
    print("GLRLM Features Extraction Began")
    for idx in range(len(normal_images)):    normal_images_features[idx]    += extract_GLRLM_features(normal_images[idx])
    for idx in range(len(benign_images)):    benign_images_features[idx]    += extract_GLRLM_features(benign_images[idx])
    for idx in range(len(malignant_images)): malignant_images_features[idx] += extract_GLRLM_features(malignant_images[idx])
    print("GLRLM Features Extraction Done")
        
if GLSZM_features:
    print("GLSZM Features Extraction Began")
    for idx in range(len(normal_images)):    normal_images_features[idx]    += extract_GLSZM_features(normal_images[idx])
    for idx in range(len(benign_images)):    benign_images_features[idx]    += extract_GLSZM_features(benign_images[idx])
    for idx in range(len(malignant_images)): malignant_images_features[idx] += extract_GLSZM_features(malignant_images[idx])
    print("GLSZM Features Extraction Done")
    
if LBP_features:
    print("LBP Features Extraction Began")
    for idx in range(len(normal_images)):    normal_images_features[idx]    += extract_LBP_features(normal_images[idx])
    for idx in range(len(benign_images)):    benign_images_features[idx]    += extract_LBP_features(benign_images[idx])
    for idx in range(len(malignant_images)): malignant_images_features[idx] += extract_LBP_features(malignant_images[idx])
    print("LBP Features Extraction Done")

#%%
num_of_features = len(normal_images_features[0])
 
print("Number of features = ", num_of_features)
with open(rf"{res_filepath}/num_of_features.dat","w") as fileOut:
    str = f"Num_of_features: {num_of_features}"
    fileOut.write(str)
#%%
###############################################################################
##################### Create Images dataframes with labels ####################
###############################################################################
print("Create Images dataframes with labels Began")
ColumnsUsed = [f"feature_{i+1}" for i in range(num_of_features)]
data_df = pd.DataFrame(columns=ColumnsUsed + ["label"] )
for i in range(len(normal_images_features)): 
    data_df.loc[i] = list(normal_images_features[i]) + [0]                 # 0 -> normal
previous_len = len(normal_images_features)
for i in range(len(benign_images_features)): 
    data_df.loc[previous_len+i] = list(benign_images_features[i]) + [1]    # 1 -> benign
previous_len += len(benign_images_features)
for i in range(len(malignant_images_features)): 
    data_df.loc[previous_len+i] = list(malignant_images_features[i]) + [2] # 2 -> malignant
print("Create Images dataframes with labels Done")
#%%
###############################################################################
########################### Split dataset train-test ##########################
###############################################################################
X        = pd.DataFrame(data_df[ColumnsUsed], columns=ColumnsUsed)
y        = data_df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, shuffle=True, random_state=42)  
preprocess_pipeline = Pipeline_preprocessor([], ColumnsUsed)   

#%%
###############################################################################
####################### Data cleaning & transformations #######################
###############################################################################   
print("Data cleaning & transformations Began")                                                                                    
X_train  = pd.DataFrame(X_train, columns=ColumnsUsed)
y_train  = pd.Series(y_train, name="label")
y_train.index = [i for i in range(len(y_train))]
X_train  = pd.DataFrame(preprocess_pipeline.fit_transform(X_train), columns=ColumnsUsed)     #X_train_prepared from now on
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
########################### Gaussian RBF SVM kernel ###########################
###############################################################################
rbf_kernel_svm_clf = SVC(kernel="rbf",gamma="scale",C=1.0)
clf_txt            = "2 Gaussian RBF SVM"
rbf_kernel_svm_clf, score = Classifier_Fit(rbf_kernel_svm_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
################################ Polynomial SVM ###############################
###############################################################################
pol  = PolynomialFeatures(degree=2)
Xpol = pol.fit_transform(X_train)
pol_svm_clf = LinearSVC(C=1, loss="hinge")
clf_txt     = "3 Polynomial SVM"
pol_svm_clf, score = Classifier_Fit(pol_svm_clf, Xpol, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
################################ Random Forest ################################
###############################################################################
rnd_clf = RandomForestClassifier(n_estimators=200, max_features="sqrt", max_depth=10, max_leaf_nodes=None, n_jobs=-1)
clf_txt = "4 Random Forest"
rnd_clf, score = Classifier_Fit(rnd_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
feature_importance = rnd_clf.feature_importances_
#%%
###############################################################################
################################## Ada Boost ################################## 
###############################################################################
ada_clf = AdaBoostClassifier()
clf_txt = "5 Ada Boost"
ada_clf, score = Classifier_Fit(ada_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
############################ k-Neighbors classifier ###########################
###############################################################################
kNeighbor_clf  = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
clf_txt = "6 k-Neighbors"
kNeighbor_clf, score = Classifier_Fit(kNeighbor_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
models_CV_res_tab = tabulate(models_CV_res, tablefmt='grid')
with open(rf"{res_filepath}/models_CV_res.dat","w") as fileOut:
    fileOut.write(models_CV_res_tab)
#%%
###############################################################################
############################### Model Selection ###############################
###############################################################################
if RunModelSelection == True:
    t1 = time.time()
    msg = "######## Executing models hyperparameters selection ########"
    print(msg)
    cv_scores = {}
    for model_name, params in model_parameters.items():
        grid_search = GridSearchCV(params["model"], params["params"], cv=Cv)
        if model_name == "pol_svm":
            grid_search.fit(Xpol, y_train)
        else:
            grid_search.fit(X_train, y_train)
        cv_scores[model_name]  = [grid_search.best_params_ , grid_search.best_score_]
        models_hyperparameters = cv_scores
        models_hyperparameters_df = pd.DataFrame.from_dict(cv_scores, orient="index", columns=["hyper_params", "Best_score"])
        grid_res = tabulate(pd.DataFrame(grid_search.cv_results_) , headers="keys", tablefmt='grid')  
        with open(rf"{res_filepath}/grid_res_{model_name}.dat","w") as fileOut:
            fileOut.write(grid_res)

    models_hyperparameters_txt = tabulate(models_hyperparameters, tablefmt="grid")
    json_object = json.dumps(cv_scores, indent=4)
    with open(rf"{res_filepath}/models_hyperparams.json", "w") as outfile:
        outfile.write(json_object)
        
    t2 = time.time()
    msg = f"Model Selection {round((t2-t1)/60,2)}mins"
    msg +="\n######### End of models hyperparameters selection #########"
    print(msg)
else:
    with open(rf"{res_filepath}/models_hyperparams.json", "r") as inpfile:
        models_hyperparameters = json.load(inpfile)
#%%
###############################################################################
############################## Create final model #############################
###############################################################################
pol_svm = LinearSVC(**models_hyperparameters["pol_svm"][0], random_state=42)
pol_svm.fit(Xpol, y_train)
#%%
lin_svm = LinearSVC(**models_hyperparameters["lin_svm"][0], random_state=42)
lin_svm.fit(X_train, y_train)
#%%
gnd_clf = RandomForestClassifier(**models_hyperparameters["gnd_clf"][0], random_state=42)
gnd_clf.fit(X_train, y_train)
#%%
rbf_svm = SVC(**models_hyperparameters["rbf_svm"][0], kernel="rbf", random_state=42)
rbf_svm.fit(X_train, y_train)
#%%
final_model_clf = VotingClassifier(estimators=[
                                     ("pol_svm",  pol_svm),
                                     ("lin_svm",  lin_svm),
                                     ("gnd_clf",  gnd_clf),
                                    ], voting="hard")  
final_model_clf.fit(X_train, y_train) 
#%%
###############################################################################
################################ Test Set scores ##############################
###############################################################################
Test_Set_Results = {}
X_test_prepared = pd.DataFrame(preprocess_pipeline.transform(X_test), columns=ColumnsUsed)                                                                              
for clf_name, clf in [("pol_svm", pol_svm), ("lin_svm", lin_svm), ("gnd_clf", gnd_clf),("rbf_svm", rbf_svm), ("final_model", final_model_clf)]:
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
                                          labels=["normal","benign","malignant"],
                                          cMap="default",Rotations=[0.,0.],
                                          savePlot=[saveDiagrams,fig_filepath,"Confusion_Matrix"]) 
