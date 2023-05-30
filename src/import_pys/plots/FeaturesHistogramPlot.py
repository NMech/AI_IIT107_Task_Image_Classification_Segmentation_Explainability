# -*- coding: utf-8 -*-
from   BasicFigureTemplate import BasicFigure_Template 
from   scipy.stats         import norm
import matplotlib.pyplot   as plt
#%%
class FeaturesHistogramPlot(BasicFigure_Template):
    
    def __init__(self, FigureProperties=["a3paper","pdf","landscape","white",0.5], FontSizes=[20.0,16.0,14.0,10.0]):
        """
        Initialization.\n
        """
        BasicFigure_Template.__init__(self, FigureProperties, FontSizes)
        self.nbins = 20
        self.alpha = 0.3
        
    def HistPlot(self, data, class_col, savePlot=["False","<filepath>","<filename>"]):
        """
        Keyword arguments:\n
            data : [pd.DataFrame].\n
        """
        mu, std   = norm.fit(data)
        columns   = list(data.columns)
        nFigs     = len(columns)-1
        nx, ny, Idxs = self.Decide_nFigs(nFigs)
        
        dim1,dim2 = self.FigureSize()   
        fig, ax   = plt.subplots(nx, ny, figsize=(dim1,dim2))

        for ix, iy in Idxs:
            if nx == 1 and ny == 1:
                data.groupby(class_col)[columns[ix+nx*iy]].hist(bins=self.nbins, alpha=self.alpha, ax=ax, edgecolor="black", legend=True, density=1)
                ax.set_xlabel(columns[ix+nx*iy])
            elif nx == 1 and ny != 1:
                data.groupby(class_col)[columns[ix+nx*iy]].hist(bins=self.nbins, alpha=self.alpha, ax=ax[iy], edgecolor="black", legend=True, density=1)
                ax[iy].set_xlabel(columns[ix+nx*iy])
            else:
                data.groupby(class_col)[columns[ix+nx*iy]].hist(bins=self.nbins, alpha=self.alpha, ax=ax[ix, iy], edgecolor="black", legend=True, density=1)
                ax[ix, iy].set_xlabel(columns[ix+nx*iy])
                              
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig,savePlot[1],savePlot[2])
          
        return fig, ax