# -*- coding: utf-8 -*-
import seaborn           as sn
import matplotlib.pyplot as plt
import numpy             as np
from BasicFigureTemplate import BasicFigure_Template
from sklearn             import metrics
#%%
class ClassificationMetricsPlot(BasicFigure_Template):
    
    def __init__(self, yTrue, FigureProperties=["a3paper","pdf","landscape","white",0.5], FontSizes=[20.0,16.0,14.0,10.0]):
        """
        Initialization.\n
        Keyword arguments:\n
            yTrue : True labels.\n
        """
        self.yTrue   = yTrue
        self.nLabels = len(np.unique(yTrue))
        BasicFigure_Template.__init__(self,FigureProperties,FontSizes)
        
    def __metrics(self,yPred):
        """
        Auxiliary function used for calculating different metrics.
        """
        #'micro', 'macro', 'weighted'
        if self.nLabels != 2:
            aveRage = "weighted" 
        else:
            aveRage = "binary"
        accuracy = round(metrics.accuracy_score(self.yTrue,yPred),3)
        precision= round(metrics.precision_score(self.yTrue, yPred,average=aveRage),3)
        recall   = round(metrics.recall_score(self.yTrue, yPred,average=aveRage),3)
        f1       = round(metrics.f1_score(self.yTrue, yPred,average=aveRage),3)
        
        return accuracy, precision, recall, f1
    
    def Confusion_Matrix_Plot(self, yPred, CMat, normalize=False, labels="auto", cMap="default", Title="",
                                    Rotations=[0.,0.], savePlot=["False","<filepath>","<filename>"]):
        """
        Implementation of confusion matrix using seaborn's heatmap. See also metrics.ConfusionMatrixDisplay.\n
        Keyword arguments:\n
            yPred     : Predicted labels from classifier.\n
            CMat      : Confusion matrix calculated from sklearn.metrics.confusion_matrix.\n
            normalize : Boolean. If True then the values of the confusion matrix are normalized.\n
            labels    : Labels of the classes. By default "auto".\n
            cMap      : Cmap to be used.\n
            Title     : Title used in the plot.\n
            Rotations : x,y-ticks rotations. Default values 0. and 0. degrees.\n
            savePlot  : list conatining the following.\n
                        * Save plot boolean.\n
                        * Filepath where the diagram will be saved.\n
                        * Filename (without the filetype) of the diagram to be plotted.\n
        Returns fig, ax.
        """
        if normalize == True:
            CMat = CMat/sum(sum(CMat))*100.
        if cMap == "default":
            cMap = plt.cm.Blues
            
        dim1,dim2   = self.FigureSize()   
        fig,ax      = plt.subplots(figsize=(dim1,dim2))
        sn.heatmap(CMat,annot=True,linewidths=.5,linecolor="black",xticklabels=labels,
                   yticklabels=labels,cmap=cMap,fmt=".4g")
        ax.tick_params(axis="x",rotation=Rotations[0])
        ax.tick_params(axis="y",rotation=Rotations[1])
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
   
        accuracy, precision, recall, f1 = self.__metrics(yPred)
        Text = f"Accuracy :{accuracy}\nPrecision :{precision}\nRecall      :{recall}\nf1            :{f1}"
        fig.text(0.90,0.90,Text,fontsize=11)
        fig.suptitle(Title)
        
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig,savePlot[1],savePlot[2])
          
        return fig, ax
    
    def Precision_Recall_Plots(self, yPred, yscores, savePlot=["False","<filepath>","<filename>"]):
        """
        Precision (Positive predictive value) vs recall (sensitivity/True positive rate) diagram.\n
            * :math:`PPV=\dfrac{TP}{TP+FP}` 
            * :math:`TPR=\dfrac{TP}{TP+FN}`
        Keyword arguments:\n
            yscores : Confidence/probability scores of predictions.\n
        Returns fig, ax.
        """
        precision = metrics.precision_score(self.yTrue, yPred)
        precisions, recalls, thresholds = metrics.precision_recall_curve(self.yTrue, yscores)
        idxs = np.where(precisions>=precision)
        
        dim1,dim2   = self.FigureSize()   
        fig,ax      = plt.subplots(figsize=(dim1,dim2))
        ax.plot(thresholds, precisions[:-1], linestyle="-", color="k", label="Precision")
        ax.plot(thresholds, recalls[:-1], linestyle="-", color="r", label="Sensitivity")
        ax.plot([thresholds[idxs[0][0]], thresholds[idxs[0][0]]], [0, 1], linestyle="--", color="b", label="current threshold")                                                                                                                       
        ax.set_xlim([min(thresholds),max(thresholds)])
        ax.set_ylim([0., 1.])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("[-]")
        ax.grid()
        ax.legend()       
        
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig,savePlot[1],savePlot[2])
        
        return fig, ax
      
    def ROC_Plot(self, yscores, savePlot=["False","<filepath>","<filename>"]):
        """
        Receiver Operating Characteristic (ROC) curve diagram.\n
        False positive rate (1-specificity) vs True positive rate (Recall).\n
            * :math:`FPR=\dfrac{FP}{N}`
            * :math:`TPR=\dfrac{TP}{TP+FN}`
        ROC curves can only be used to assess classifiers that return some\n
        confidence score (or a probability) of prediction.\n
        Some classifiers:\n
            * Logistic regression.\n
            * Decision trees/Random Forests boosting, bagging etc.\n
            * Neural networks.\n
        Keyword arguments:\n
            yscores : Confidence/probability scores of predictions.\n
        Returns fig, ax.
        """
        fpr, tpr, thresholds = metrics.roc_curve(self.yTrue,yscores)
        dim1,dim2 = self.FigureSize()   
        fig,ax    = plt.subplots(figsize=(dim1,dim2))
        ax.plot(fpr,tpr,linestyle="-",color="k")
        ax.plot([0.,1.], [0.,1],linestyle="--",color="k")
        ax.set_xlabel("False Positive Rate (1-specificity)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_xlim([0.,1.])
        ax.set_ylim([0.,1.])
        ax.grid()
        auc_score = round(metrics.roc_auc_score(self.yTrue,yscores),3)
        Text = f"AUC score : {auc_score}"
        fig.text(0.88,0.20,Text,fontsize=11)    
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig,savePlot[1],savePlot[2])
        
        return fig, ax