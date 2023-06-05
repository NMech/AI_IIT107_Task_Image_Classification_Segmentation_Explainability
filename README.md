# AI_IIT107_Task_Image_Classification_Segmentation_Explainability
## Introduction
We investigate the application of machine learning and deep learning techniques for classifying and segmenting breast tumour ultrasound images. The classification task focuses on categorizing tumours into three classes: normal, benign, and malignant, aiding radiologists in the initial assessment of tumour malignancy. Simultaneously, the segmentation task aims to identify cancerous regions within the images, regardless of their specific malignant characteristics, facilitating targeted analysis and treatment planning.

## Basic Code Information
The code has been developped in **python 3.8.2**. In order to run the code, you will need to use the ml_requirements.txt/dl_requirements.txt files to install all the necessary modules and packages. Once installed, you can run the code using an integrated development environment (IDE) such as Spyder. Simply execute one of the four main files (ml_classification.py, ml_segmentation.py, dl_classification.py, dl_segmentation.py) located inside the src folder.

## Repository Structure
The repository structure is simple and self-explanatory. It containts the following folders and files:

**ml_requirements.txt** - File that contains all the modules/packages information needed to run the code for the Machine Learning tasks.

**dl_requirements.txt** - File that contains all the modules/packages information needed to run the code for the Deep Learning tasks.

**Presentation folder** - Contains the presentation both in .pptx and .pdf format.

**Report folder** - Contains the report as .pdf file. Also, there is a compressed file that contains the LaTeX code that has been created.

**results folder** - Contains all the results that the code produces (.dat and .json).

**models folder** - Contains all the exported trained models.

**src folder** - contains the following files and folders
| Files/Folders         |  Description                                      |               
|-----------------------|---------------------------------------------------|
| ml_classification.py  | Main file of Machine Learning Classification task |
| ml_segmentation.py    | Main file of Machine Learning Segmentation task   |
| dl_classification.py  | Main file of Deep Learning Classification task    |
| dl_segmentation.py    | Main file of Deep Learning Segmentation task      |
| demo.py               | File that contains the code of our demo. Demo is only used for the Machine Learning Segmentation Task          |
| import_pys            | Code has been splitted in multiple files for better overview. This folder contains all the .py files apart from the main ones |

## Basic Results
Below we can see the results of our trained for each of our tasks. More details can be found in the report.

### Machine Learning Classification
| Metric    | Voting Classifier             | 
|-----------|-------------------------------|
| Accuracy  | 0.7308                        |
| Precision | 0.6428,    0.7212,    0.7895  |
| Recall    | 0.3214,    0.9146,    0.6521  | 
| F1 score  | 0.4286,    0.8064,    0.7143  |

### Machine Learning Segmentation
| Metric    | Voting Classifier | 
|-----------|-----------------|
| Accuracy  | 0.7906          |
| Precision | 0.8272, 0.7638  |
| Recall    | 0.7183, 0.8586  | 
| F1 score  | 0.7689, 0.8085  |

### Deep Learning Classification
| Metric    | Voting Classifier    | 
|-----------|----------------------|
| Accuracy  | 0.796                |
| Precision | 0.71,  0.857, 0.738  |
| Recall    | 0.815, 0.818, 0.738  | 
| F1 score  | 0.759, 0.837, 0.738  |

### Deep Learning Segmentation
| Metric    | Voting Classifier | 
|-----------|-----------------|
| Accuracy  | 0.957         |
| Precision | 0.973, 0.721  |
| Recall    | 0.973, 0.721  | 
| F1 score  | 0.973, 0.721  |
