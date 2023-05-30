# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm      import SVC, LinearSVC
#%%
model_parameters = {         
    "gnd_clf": # Random Forest
    {
        "model":RandomForestClassifier(random_state=42),
        "params":
        {
            "n_estimators":[50, 100, 200],
            "max_depth"    :[3, 6, 8, 10, 12],
            "max_features":["auto","sqrt","log2"],
        }
    },
        
    "pol_svm": # Polynomial SVM
        {
            "model" :LinearSVC(),
            "params":
            {
                "loss"  : ["hinge", "squared_hinge"],
                "C"     : [0.1, 0.5, 1.0]
            }
        },
            
    "lin_svm": # Linear SVM
        {
            "model" :LinearSVC(),
            "params":
            {
                "loss"  : ["hinge", "squared_hinge"],
                "C"     : [0.1, 0.5, 1.0]
            }
        },
            
    "rbf_svm": # Gaussian RBF SVM
        {
            "model" : SVC(kernel="rbf", random_state=42),
            "params":
            {
                "gamma" : ["scale", "auto"],
                "C"     : [0.1, 0.5, 1.0]
            }
        }         
}