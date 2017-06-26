# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:07:22 2016

@author: kevin
Preprocess images
"""
import open_save_images as imgs
import os
import numpy as np
import scipy.misc
from scipy import ndimage
from skimage import transform as tf
import gc
import random
from shutil import copytree, move
import pandas as pd
#files = imgs.getFiles("convNetPanelsPaneled_sorted/bar/")
class imagePreprocess(object):

    images = []
    shapes = []
    glob_n = None
    glob_q = None
    def __init__(self):
        global glob_n,glob_q
        glob_n = 106
        glob_q = 106
        pass
    '''
    1) make directory outside of the images directory and copy images
    2) Move some images for the test set
    3) resize, shift, rotate, shear
    '''
    def initialize_images(self, directory, n_images = 10):
        #1)
        print 'Copying Files'
        modified_dir = directory.replace(directory.split('/')[-1], "") + "Modified_Image_Dir"
        copytree(src=directory, dst=modified_dir)
        print 'Files copied'
        
        #2)
        print 'Making file list'
        test_dir = directory.replace(directory.split('/')[-1], "") + "Test_Image_Dir/"
        df = pd.DataFrame(columns=('path', 'class'))
        file_list = []
        for path, subdirs, files in os.walk(modified_dir):
            for name in files:
                file_path = os.path.join(path, name)#can create a list of classes that we want here...
                file_list.append((file_path, file_path.split('/')[-2], file_path.split('/')[-1]))
        df = pd.DataFrame(data = file_list, columns=('path', 'class', 'filename'))
        print 'Resizing images'
        '''Resize all the images'''
        for img in df.iterrows():
            loaded_img = imgs.load_image(img[1]['path'])
            resized_img = scipy.misc.imresize(loaded_img, (glob_n, glob_q))
            imgs.save_image(resized_img, img[1]['path'])
            
        test_images_dfs = []
        for img_class in set(df['class']):
            top_N = df[df['class'] == img_class].head(n_images)
            df = df.drop(top_N.index)
            test_images_dfs.append(top_N)
        test_images = pd.concat(test_images_dfs)
        print 'Moving Test Images'
        for img in test_images.iterrows():
            subdir = os.path.join(test_dir, img[1]['class'])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            move(img[1]['path'], os.path.join(test_dir, img[1]['class'], img[1]['filename']))
            test_images.set_value(img[0],'path', os.path.join(test_dir, img[1]['class'], img[1]['filename']))
        
        print 'Saving to seperate test and train variables'
        x_train = df['path']
        y_train = df['class']
        x_test = test_images['path']
        y_test = test_images['class']
        return (x_train, y_train, x_test, y_test)
        
    
    def modify_images(df):
        print 'Performing: shift, rotate, shear'
        for img in df.iterrows():
            loaded_img = imgs.load_image(img[1]['path'])
            for r in range(-10,11):
                if r != 0:
                    imgs.save_image(self.rotateImage(loaded_img,r), img[1]['path'].replace('.png', '_ROTATE_' + str(r) + '.png'))#append each set of images
                    imgs.save_image(self.shiftImage(loaded_img,r,0), img[1]['path'].replace('.png', '_SHIFT_' + str(r) + '.png'))
                    imgs.save_image(self.shearImage(loaded_img,r/100.), img[1]['path'].replace('.png', '_SHEAR_' + str(r) + '.png'))
    
    def GetN_Q(self, images, scaleFactor):
        #axis means make nxn using width or height
        m = 0#Shortest Side
        l = 0#longest side
        width = 0
        height = 0
        lenImages = len(images)#number of images
        for i in range(lenImages):
            width = shapes[i][0]
            height = shapes[i][1]        
            if width < height:
                m += width
                l += height
            else:
                m += height
                l += width
        #calculate average m and l    
        m /= lenImages
        l /= lenImages
        
        #scale down the average
        n = m /scaleFactor
        q = l / scaleFactor
        global glob_n,glob_q
        glob_n = n
        glob_q = q
        #print(n)
    
    def normalizeImageSize(self, images, scaleFactor):
        global glob_n
        global glob_q
        global shapes
        NewImgs = []
        lenImages = len(images)
        for i in range(lenImages):
            width = shapes[i][0]
            height = shapes[i][1] 
            if width < height:#q is the width
                NewImgs.append(scipy.misc.imresize(images[i], (glob_n, (glob_q+glob_n)/2),'cubic'))
            else:#q is the height
                NewImgs.append(scipy.misc.imresize(images[i], ((glob_q+glob_n)/2, glob_n),'cubic'))
                
        return NewImgs
    def rotateImage(self, image, deg, reshape = False):
        #reshape means if you want to reshape the image        
        return ndimage.rotate(image, deg, reshape=reshape)
        
    
    def shiftImage(self, image, shiftSizeX,shiftSizeY):
        if len(np.shape(image)) == 2:
            return ndimage.shift(image, (shiftSizeY, shiftSizeX),mode='nearest')
        else:
            return ndimage.shift(image, (shiftSizeY, shiftSizeX, 0),mode='nearest')
    
    def shearImage(self, image, shear):
        #reshape means if you want to reshape the image
        afine_tf = tf.AffineTransform(shear=shear)
        return tf.warp(image, afine_tf)
        
    def centerImage(self, images):#you need an n value otherwise it will set it to 0 and do nothing
        NewImgs = []
        global glob_n,glob_q
        for i in range(len(images)):
            width = np.shape(images[i])[1]
            height = np.shape(images[i])[0]
            if width < height:
                center = height/2
                #print("CenterHeight")            
                #print(center, height, i, len(range(center-int(round(glob_n/2.)),center+int(round(glob_n/2.)))))
                NewImgs.append(images[i][center-int(round(glob_n/2.)):center+int(round(glob_n/2.)),range(glob_n)])
            else:
                center = width/2
               # print("CenterWidth" + str(glob_n) + str(np.shape(images[i])) + 
                #"ShapeAfter:" + str(np.shape(images[i][range(glob_n),center-int(round(glob_n/2.)):center+int((glob_n/2.))])))
                #print(center, width, len(range(center-int(round(glob_n/2.)),center+int(round(glob_n/2.)))))
                
                NewImgs.append(images[i][range(glob_n),center-int(round(glob_n/2.)):center+int((glob_n/2.))])
        return NewImgs

#
#images2 = normalizeImageSize(images, 2)
#images2 = centerImage(images2)
#os.mkdir( "normalized", 0777 );
#for i in range(10):
#    imgs.save_image(images2[i],"normalized/out"+ str(i) +".png")
#
#imgs.save_image(images2[0],"/home/kevin/out[0]"+ str(0) +".png")
#width = (np.shape(images2[0])[0])/2
#imgs.save_image(images2[0][width-glob_n:width+glob_n,],"/home/kevin/out[0]"+ str(0) +".png")
#
#imgs.save_image(images[0],"/home/kevin/out[1]"+ str(0) +".png")
#width = (np.shape(images[3])[1])/2
#imgs.save_image(images[3][:,width-glob_n:width+glob_n],"/home/kevin/out[1]"+ str(1) +".png")
#
#images3 = rotateImage(images, 5, False)
#imgs.save_image(images3[0],"/home/kevin/out.png")
#
#images4 = shiftImage(images, 5, 0)
#imgs.save_image(images4[1],"/home/kevin/out.png")
#
#images5 = shearImage(images, -0.02)
#imgs.save_image(images5[1],"/home/kevin/out.png")

