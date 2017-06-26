#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:23:20 2017

@author: kevin
"""
import os
from xml.dom import minidom
from bs4 import BeautifulSoup
import pandas as pd
from glob import glob
import urllib2
from HTMLParser import HTMLParser
import re
#import lucene
#from org.apache.lucene.index import IndexOptions
#from org.apache.lucene.store import IOContext
#from org.apache.lucene.store import Directory
#from org.apache.lucene.analysis.standard import StandardAnalyzer
#from org.apache.lucene.document import Document, Field, FieldType
#from org.apache.lucene.search import IndexSearcher
#from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexReader, DirectoryReader
#from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser, QueryParserBase
#from org.apache.lucene.store import SimpleFSDirectory
#from org.apache.lucene.util import Version
#from java.io import File
#os.chdir('/home/kevin/Downloads/ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/06/00')




def untar(fname):
    import tarfile
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()


def get_image_information(data):
    columns=['title','label', 'caption', 'image_name']
    image_list = []
    title = data.getElementsByTagName('article-title')[0].firstChild.nodeValue
    for idx, fig in enumerate(data.getElementsByTagName("fig")):
        #print fig.getAttribute('id')
        label = fig.getElementsByTagName('label')[0].firstChild.nodeValue#label
        caption = fig.getElementsByTagName('caption')[0].firstChild.firstChild.nodeValue#caption
        imageName = fig.getElementsByTagName('graphic')[0].getAttribute('xlink:href')#.jpg
        image_list.append({'title':title, 'label': label, 'caption' : caption, 'image_name': imageName})
    images = pd.DataFrame(columns = columns, data = image_list)
    return images


def get_image_caption(data):
    for row in data:#every row in the dataframe
        get_nxml(fig_df['PMCID'])
    return images

def get_nxml(pmcid):
    f = urllib2.urlopen('https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:' + str(pmcid) + '&metadataPrefix=pmc')
    blob = f.read()
    data = BeautifulSoup(blob, 'xml')
    return data

def get_image_PMCID_Fig(files):    
    '''
    This will give us the PMCID and the figure number so we can get the xml
    for the file and then we can go into the xml and get the caption for the 
    figure
    Input: list of files in an array
    Output dataframe of files, along with locations, figure number
    '''
    import re
    #.(?=_process)
    files_w_fig = []
    for img in files:
        
        path = img
        try:
            fig_num = re.search(r'[1-9]+(?=_process)', img).group()
        except:
            fig_num = -1#we dont know the figure number?
        PMCID = re.search(r'(?<=PMC)(.*)(?=__)', img).group()
        img_class = img.split('/')[-2]
        figure_id = re.search(r'.+(?=_process)', img.split('/')[-1]).group().replace("__", "/")
        
        files_w_fig.append((path, fig_num, PMCID, img_class, figure_id))#get the figure number
    result = pd.DataFrame(data = files_w_fig, columns = ['path','figure_num', 'PMCID', 'class', 'figure_id'])
    #result['caption'] = pd.Series('', index=result.index)
    #result['nxml'] = pd.Series('', index=result.index)
    return result

def find_caption(figure_number):
    caption = None
    try:
        caption = str(nxml.find(id='F'+figure_number).caption)
    except:
        try:
            caption = str(nxml.find(id='fig'+figure_number).caption)
        except:
            caption = ""

    return caption


    
if __name__ == '__main__':
    '''
    If the caption is the same then this means that it is from the same image.
    '''

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
    
    #Start Session
    session = Session(engine)
    
    working_directory = '/home/kevin/Downloads/Lab/convNetPanelsPaneled_sorted/'
    #get the tar.gz files recursivly within the working_dir
    results = [y for x in os.walk(working_directory) for y in glob(os.path.join(x[0], '*.png'))]
    
    fig_df = get_image_PMCID_Fig(results)
    
    for fig in fig_df.iterrows():
        fig = fig[1]
        session.add(images(path=fig['path'], pmcid=fig['PMCID'], img_class=fig['class'], fid=fig['figure_id']))
    session.commit()
    
    '''
    We use this method to store the figids into a file
    '''
    
    results_fig = [re.search(r'.+(?=_process)', img.split('/')[-1]).group().replace("__", "/") for img in results ]
    import pickle
    with open("results_fig.txt", "wb") as fp:   #Pickling
        pickle.dump(results_fig, fp)
    
        
    #split into the directory and filename
    #results = [tuple(x.rsplit('/',1)) for x in results]    
    #untar the results
    
    
    
    
    '''
    #This is for getting the captions directly from pubmed, we dont need this since
    #we have the captions from the index
    
    set_PMCID = list(set(fig_df['PMCID']))
    len_list = len(set_PMCID)
    len_images = len(fig_df)
    counter = 0
    for i, pmcid in enumerate(set_PMCID):
        nxml = get_nxml(pmcid)
        figures = fig_df[fig_df['PMCID'] == pmcid]
        #add a row:
        session.add(full_text(pmcid = pmcid, text = str(nxml)))
        #session.commit()

        for idx, fig in figures.iterrows():
            counter += 1 
            #print type(fig)
            #fig_df.ix[fig.name,'caption']= re.sub('<[^<]+?>', '', str(nxml.find(id='F'+fig.figure_num).caption))
            fig_df.set_value(fig.name, 'caption', re.sub('<[^<]+?>', '', find_caption(fig.figure_num)))
            session.add(images(pmcid = pmcid, path=fig_df.iloc[fig.name]['path'], caption=fig_df.iloc[fig.name]['caption'], img_class = fig_df.iloc[fig.name]['class']))
            #session.commit()
        print str(i) + '/' + str(len_list) + ' pmcid: ' + pmcid
        if i % 20 == 0:
            session.commit()
            print "image {} out of {}".format(counter, len_images)
               
    session.commit()
    '''
    