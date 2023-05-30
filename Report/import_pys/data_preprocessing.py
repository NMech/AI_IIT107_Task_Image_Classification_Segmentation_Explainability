# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose       import ColumnTransformer
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
#%%
def Pipeline_preprocessor(cat_attribs, num_attribs):
    """
    OneHotEncoder in categorical data columns.\n
    StandardScaler in numerical data columns.\n
    Passthrough remaining columns.\n
    """
    num_transformer = Pipeline(steps=[
        ("imputer"    , SimpleImputer(strategy="median")),
        ("std_scaler" , StandardScaler())
        ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer"   , SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot"    , OneHotEncoder(handle_unknown="ignore"))
        ])
    
    col_trans = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_attribs),
            ("cat", cat_transformer, cat_attribs),
            ],
        remainder="passthrough")

    full_pipeline = Pipeline(steps=[("col_trans", col_trans)])

    return full_pipeline