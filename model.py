#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:49:25 2017

@author: kevin
"""
import os
os.chdir('/media/kevin/Anime/tmpDL/Lab')
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Input, Flatten
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.utils.np_utils import to_categorical
from keras.layers.merge import Concatenate
from keras.models import Model  
#from gensim.models import Word2Vec
import numpy as np
import imagePreprocess
import open_save_images as imgs
import pandas as pd
import re
from keras.callbacks import ModelCheckpoint
IMAGE_DIRECTORY = '/media/kevin/Anime/tmpDL/Lab/convNetPanelsPaneled_sorted'
CAPTION_CSV_LOCATION = 'fid_caprtion.csv'
IMAGE_X = 106
IMAGE_Y = 106

LOAD_IMAGES_FROM_DISK = False
LOAD_CAPTIONS_FROM_DISK = True
'''Get top 2000 words'''
NUM_WORDS = 2000
def get_captions_db():
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine
    
    Base = automap_base()
    
    # engine, suppose it has two tables 'user' and 'address' set up
    engine = create_engine('mysql+mysqldb://root:toor@127.0.0.1/PubMedArticles', pool_recycle=3600) # connect to server
    
    # reflect the tables
    Base.prepare(engine, reflect=True)
    
    # mapped classes are now created with names by default
    # matching that of the table name.
    images = Base.classes.images
    fid_caption = Base.classes.fid_caption
    #Start Session
    session = Session(engine)    
    #innerjoin on fid, since we have images with the same fid but different file names
    results = session.query(fid_caption, images).join(images, images.fid == fid_caption.fid).all()
    
#    for img in results:
#        print img.fid_caption.caption#call the result.table.column
#        break
        
#    for img in q:
#        print "This is the pmcid {},\n This is the img path {},\n this is the img caption {}.".format(img.pmcid,img.path, img.caption)
    return results

def get_captions(fid_img):
    if LOAD_CAPTIONS_FROM_DISK:
        return CAPTION_CSV[CAPTION_CSV['fid'] == fid_img].caption.iloc[0]
    else:
        results = session.query(fid_caption).filter_by(fid = fid_img).all()        
        for q in results:
            return q.caption

def preprocess_images(save = True):
    #preprocess images:
#    classes_text = ['box', 'network', 'fluorescence', 'gel', 'topology', 'pie', 'histology',
#               'text', 'map', 'heatmap', 'photo', 'plot', 'screenshot', 'table', 'tree',
#                'molecular', 'sequence', 'medical', 'diagram', 'microscopy', 'line', 'bar']
    classes_text = ["bar","gel","histology","line","molecular","network","plot","sequence"]
    ipp = imagePreprocess.imagePreprocess()
    
    (x_train, y_train, x_test, y_test) = ipp.initialize_images(IMAGE_DIRECTORY, 10)
    x_train = x_train.to_frame()
    x_test = x_test.to_frame()
    x_train['caption'] = pd.Series('', index=x_train.index)
    x_train['fid'] = pd.Series('', index=x_train.index)
    #x_train['image'] = pd.Series('', index=x_train.index)
    x_test['caption'] = pd.Series('', index=x_test.index)
    x_test['fid'] = pd.Series('', index=x_test.index)
    #x_test['image'] = pd.Series('', index=x_test.index)
    
    #y_train values to numerical
    print 'set y values to one hot array'
    y_train_categorical = []
    for img_class in y_train:
        y_train_categorical.append(classes_text.index(img_class))
    y_train = y_train_categorical
    #y_test values to numerical
    y_test_categorical = []
    for img_class in y_test:
        y_test_categorical.append(classes_text.index(img_class))
    y_test = y_test_categorical
    
    #add fid, caption, and image array to each of the rows in the training and testingset
    
    for record in x_train.iterrows():
        x_train.set_value(record[0], 'fid', re.search(r'.+(?=_process)', record[1]['path'].split('/')[-1]).group().replace("__", "/"))
        x_train.set_value(record[0],'caption', get_captions(record[1]['fid']))
    
    
    for record in x_test.iterrows():
        x_test.set_value(record[0], 'fid', re.search(r'.+(?=_process)', record[1]['path'].split('/')[-1]).group().replace("__", "/"))
        x_test.set_value(record[0],'caption', get_captions(record[1]['fid']))
    
    #set to Categorical
    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test)
    if save:#if we want to save the arrays
        x_train.to_pickle('x_train.pkl')
        np.save('y_train', y_train)#dump them as ndArrays
        x_test.to_pickle('x_test.pkl')
        np.save('y_test', y_test)#dump them as ndArrays
    return (x_train, y_train, x_test, y_test)

#we have to load the actual images
def load_images(df):
    ld_images = []
    for record in df.iterrows():
        img = imgs.load_image(record[1]['path'])
        if len(np.shape(img)) == 3:
            img = img[:,:,1]
        ld_images.append(img/255.0)
    return ld_images

def get_CNN():
    CNN_Model = Sequential()
    #block 1
    CNN_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(106, 106, 1), name='block1_conv1'))
    CNN_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv2'))
    CNN_Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
    
    #block 2
    CNN_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv1'))
    CNN_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv2'))
    CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    #block 3
    CNN_Model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv1'))
    CNN_Model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv2'))
    CNN_Model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv3'))
    CNN_Model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv4'))
    CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
    #block 4
    CNN_Model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv1'))
    CNN_Model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv2'))
    CNN_Model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv3'))
    CNN_Model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv4'))
    CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    CNN_Model.add(Flatten())
    return CNN_Model

def get_rnn():
    #model
    RNN_Model = Sequential()
    
    RNN_Model.add(Embedding(NUM_WORDS, 256, input_length=200))
    RNN_Model.add(Dropout(0.5))
    
    RNN_Model.add(LSTM(256,return_sequences=True,activation='relu', name='block1_lstm1'))
    RNN_Model.add(Dropout(0.5))
    RNN_Model.add(LSTM(256, return_sequences=False, activation='relu',name='block1_lstm2'))
    RNN_Model.add(Dropout(0.25))
    return RNN_Model
'''
Introduce the RNN for the text portion
'''

def tokenize_words(df):
    '''Get top 2000 words'''
    num_words = 2000
    tok = Tokenizer(num_words)
    '''Find top most used words'''
    tok.fit_on_texts(df['caption'])
    '''store out the top used words'''
    words = []
    for iter in range(num_words):
        words += [key for key,value in tok.word_index.items() if value==iter+1]
    '''print the top 10 words'''
    #words[:10]#top 10 words
    #training and testing dataset
    '''
    convert the text to numerical arrays. Literally it just takes the entries of the
    tokenizer word_index (a dictionary where the key is the word, and the value is the number)
    then matches the word with the number in the index. This index is a ranked list
    of the most popular words. In our case the it is the top 2000 words 
    (this list contains all of the words, but it is respected when we do
    texts_to_sequences)
    
    The first sentence we have: 
    Abundance (a) and taxa richness (b) of aquatic macroinvertebrates 
    (log(x\xc2\xa0+\xc2\xa01)-transformed number of individuals per square metre 
    and number of taxa, respectively). Asterisks indicate significant 
    (P\xc2\xa0<\xc2\xa00.05, ANOVA, Games\xe2\x80\x93Howell post hoc tests) 
    differences from the controls. Vertical dashed lines show contamination events
    
    It looks like this after texts_to_sequences on them
    
    1594, 5, 3, 12, 2, 814, 1355, 70, 2, 1050, 152, 844, 3, 70, 2, 97, 557, 77, 
    115, 1426, 793, 53, 559, 270, 1372, 1632, 332, 14, 1, 262, 658, 334, 86, 154, 875
    
    These are the values for each of the words. We have less here than the actual 
    number of words from the raw text because we told the tokenizer we only want 
    the top 2000 words.
    
    Ex: macroinvertebrates wont be in the numerical vectorization because:
    print tok.word_index['macroinvertebrates']
    Out[141]: 28790
    it is not in the top 2000 words.
    
    '''
    x_caption = tok.texts_to_sequences(df['caption'])
    x_caption  = sequence.pad_sequences(x_caption,  maxlen=200)
    
    return (x_caption)

if __name__ == "__main__":
    #initialize caption csv
    if 'CAPTION_CSV' not in locals() and LOAD_CAPTIONS_FROM_DISK:
            CAPTION_CSV = pd.read_csv(CAPTION_CSV_LOCATION)
    #initialize the database connection
    if 'session' not in locals() and not LOAD_CAPTIONS_FROM_DISK:
        from sqlalchemy.ext.automap import automap_base
        from sqlalchemy.orm import Session
        from sqlalchemy import create_engine
        
        Base = automap_base()
        
        # engine, suppose it has two tables 'user' and 'address' set up
        engine = create_engine('mysql+mysqldb://root:toor@127.0.0.1/PubMedArticles', pool_recycle=3600) # connect to server
        
        # reflect the tables
        Base.prepare(engine, reflect=True)
        
        # mapped classes are now created with names by default
        # matching that of the table name.
        
        fid_caption = Base.classes.fid_caption
        #Start Session
        session = Session(engine)
        #select the caption where the fid == the fid from the df

    #sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    if LOAD_IMAGES_FROM_DISK:
        x_train = pd.read_pickle('x_train.pkl')
        x_test = np.load('x_test.pkl')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
    else:
        (x_train, y_train, x_test, y_test) = preprocess_images(True)
    
    print 'Getting small sample'
    #small sample
    random_sample = np.random.randint(10, len(x_train)-1, size=7000)#set this
    x_train = x_train.iloc[random_sample]
    y_train = y_train[random_sample]
    
    print 'loading images as numpy arrays'
    x_train_images = load_images(x_train)
    x_test_images = load_images(x_test)

    print 'tokenizing captions'
    x_train_captions = tokenize_words(x_train)
    x_test_captions = tokenize_words(x_test)
    
    print 'building models'
    #get Models:
    RNN = get_rnn()
    CNN = get_CNN()
    concatOut = Concatenate()([CNN.output,RNN.output])
    model = Model([CNN.input, RNN.input], concatOut)
    final_model = Sequential()
    final_model.add(model)
    #merge model
    final_model.add(Dense(500, activation='relu', name='block1__dense1'))
    final_model.add(Dropout(0.5))
    final_model.add(Dense(250, activation='relu', name='block1__dense2'))
    final_model.add(Dropout(0.5))
    final_model.add(Dense(8, activation='softmax', name='final_dense'))
    final_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #fit model
    #we need to read in the images and make sure they are of the same size...
    x_train_images = np.array(x_train_images).reshape(len(x_train_images),IMAGE_X,IMAGE_Y,1)
    checkpoint = ModelCheckpoint('model_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    print 'Starting Training'
    final_model.fit([x_train_images,x_train_captions], y_train, batch_size=32,
                    epochs=50, verbose=1, validation_split=0.10, callbacks=callbacks_list)
    