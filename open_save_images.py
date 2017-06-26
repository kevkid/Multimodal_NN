# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:19:30 2016

@author: kevin
"""

from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.misc
import gc
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    img.close()
    del img
    #gc.collect()
    return data

def save_image( npdata, outfilename ) :
    scipy.misc.imsave(outfilename, npdata)
    
def getFiles(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles
    