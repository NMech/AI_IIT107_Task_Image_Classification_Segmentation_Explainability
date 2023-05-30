# -*- coding: utf-8 -*-
import os
import joblib
import time
import numpy as np
from   sklearn.model_selection import cross_val_score
#%%
def Classifier_Fit(classifier, X_train, y_train, Cv, clf_txt, filepathOut):
    """
    Implementation of classifier fitting.\n
    Keyword arguments:\n
        classifier : sklearn classifier object.\n
        X_train    : Training dataset [pd.DataFrame].\n
        y_train    : Labels of training dataset [pd.Series].\n 
        Cv         : Number of cross-validation folds [int].\n
        clf_txt    : Text to print in screen (also used for naming the model).\n
        filepathOut: Filepath where the models are saved.\n
    """
    t1 = time.time()
    classifier.fit(X_train, y_train)
    classifier_CV_score = cross_val_score(classifier, X_train, y_train, scoring="accuracy", cv=Cv, n_jobs=-1)
    score = round(np.mean(classifier_CV_score),3)
#    Save_Classifier(classifier, filepathOut, clf_txt)
    t2 = time.time()
    msg = f"{clf_txt:40} |{round(t2-t1,4)}s"
    print(msg)

    return classifier, score

def Save_Classifier(classifier, filepathOut, filename):
    """
    Used for saving the trained model using pickle.\n
    Keyword arguments:
        classifier : This classifier has passed from the pipeline
    """
    fileOut = os.path.join(filepathOut, filename+".joblib")
    joblib.dump(classifier, open(fileOut, "wb"))
    
    return None