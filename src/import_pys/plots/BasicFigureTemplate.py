# -*- coding: utf-8 -*-
import os
import matplotlib as mpl
#%%
class BasicFigure_Template:
    
    def __init__(self, FigureProperties=["a3paper","pdf","landscape","white",0.5], FontSizes=[20.0,16.0,14.0,14.0]):
        """
        Keyword arguments:\n
            FigureProperties: list containing the main properties of the plotted figure.\n 
                        * String defining the paperType.\n
                          Options ["a0paper","a1paper","a2paper","a3paper","a4paper","a5paper"].\n
                        * Filetype of the diagram in case it will be saved. Default value "pdf".\n
                        * Orientation of the saved diagram. Options:["landscape", "portrait"].\n
                        * Background colour. Default value "white".\n
                        * Background colour opacity. Default value 0.5.\n
            FontSizes: list containing the font sizes of the following.\n
                        * title size:20, label size:16, legend size:14, font size:14.\n
        """
        params =  {"axes.titlesize" : FontSizes[0],
                   "axes.labelsize" : FontSizes[1],
                   "legend.fontsize": FontSizes[2],
                   "font.size"      : FontSizes[3]}
        
        mpl.rcParams.update(params)
        self.FigureProperties = FigureProperties
        self.paperType = FigureProperties[0]

    def BackgroundColorOpacity(self, fig):
        """
        Define figure background colour and opacity.\n
        Returns None.
        """
        backColour = self.FigureProperties[3]
        backOpacity= self.FigureProperties[4]
        fig.patch.set_facecolor(color=backColour)
        fig.patch.set_alpha(backOpacity)
        fig.tight_layout() 
        
        return None
    
    def FigureSize(self):
        """
        Function containing dimensions (in inches) for different paper types.
        """
        paperType  = self.paperType
        paperSizes = {"a0paper":[46.8,33.1],"a1paper":[33.1,23.4],"a2paper":[23.4,16.5],
                      "a3paper":[16.5,11.7],"a4paper":[11.7,8.3] ,"a5paper":[8.3,5.8]}
        dim1,dim2  = paperSizes[paperType][0],paperSizes[paperType][1]

        return  dim1, dim2
    
    def SaveFigure(self, fig, filepath, filename):
        """
        Function used for saving a figure.\n
        Keyword arguments:\n
            fig      : plt.subplots() fig variable.\n
            filepath : Filepath of the saved diagram.\n
            filename : Filename of the saved diagram.\n
        Returns  None.
        """
        filetype = self.FigureProperties[1]
        Filename = self._CreateDirectory(filepath,filename+"."+filetype)
        fig.savefig(Filename,orientation=self.FigureProperties[2],format=filetype,facecolor=fig.get_facecolor())

        return None
    
    def Decide_nFigs(self, nFigs):
        """
        Function used for deciding the number of subplots.\n
        Keyword arguments:\n
            nFigs : Number of figures.\n
        Returns number of figures in x, y and a list of tuples with the indices\n
        of the axes to be used in the figure.\n
        """
        if nFigs > 21:
            raise Exception("More than 21 diagrams to be plotted")
            
        if nFigs in [1, 2, 3]:
            nx, ny = 1, int(nFigs)
        elif nFigs in [4, 6, 8, 10]:
            nx, ny = 2, int(nFigs/2)
        elif nFigs in [5, 7]:
            nx, ny = 2, int(nFigs/2)+1
        elif nFigs in [9, 12, 15, 18, 21]:
            nx, ny = 3, int(nFigs/3)
        elif nFigs in [11, 13, 14, 17, 19]:
            nx, ny = 3, int(nFigs/3)+1
        elif nFigs in [16, 20]:
            nx, ny = 4, int(nFigs/4)
            
        Idxs = []
        for ix in range(nx):
            for iy in range(ny):
                if ix+nx*iy >= nFigs:
                    break
                else:
                    Idxs.append((ix, iy))
        
        return nx, ny, Idxs

    def _CreateDirectory(self, filepath, filename):
        """
        Auxiliary function used for creating/moving to directories.\n
        Keyword arguments:\n
            filepath : The path of the directory to be created.\n
            filename : The name of the file to be present in the created directory.\n
        Returns the full name of the file to be handled in other methods.
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            os.chdir(filepath)
            file = os.path.join(filepath,filename)            
        else:
            os.chdir(filepath)
            file = filename
        
        return file