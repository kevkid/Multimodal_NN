#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:21:57 2017

@author: kevin
This is for lucene 4.10... Not the latest
"""

import lucene
from datetime import datetime
 
from java.io import File
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.document import Document, Field, TextField,StringField
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser, QueryParserBase
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
import re
location = "/home/kevin/Downloads/Lab/test.index"
queryString = ('figid:"PMC1312313/1471-2164-6-151-1" OR figid:"PMC122074/1472-6882-2-8-2"')
inputString = "PMC122074/1472-6882-2-8-2"

'''
Method for 2.9
'''
def reader(queryString):
    lucene.initVM()
    vm_env = lucene.getVMEnv()
    vm_env.attachCurrentThread()
    analyzer = WhitespaceAnalyzer()
    reader = IndexReader.open(SimpleFSDirectory(File(location)))
    searcher = IndexSearcher(reader)
    query = QueryParser("figid", analyzer).parse(queryString)
    
    MAX = 1000000
    hits = searcher.search(query, MAX)
    fig_w_caption = []
    if using_db == True:
        for hit in hits.scoreDocs:
            doc = searcher.doc(hit.doc)
            figid = (doc.get('figid'))
            cap = (doc.get('caption'))
            if cap is None:
                cap = 'No Caption'
            session.add(fid_caption(fid = figid.encode('utf-8'), caption = cap.encode('utf-8')))
            session.commit()
    else:
        for hit in hits.scoreDocs:
            doc = searcher.doc(hit.doc)
            print doc.get('figid')
            print doc.get('caption')

'''
Method for 4.10
'''

def writer():
    lucene.initVM()
    vm_env = lucene.getVMEnv()
    vm_env.attachCurrentThread()
    analyzer = StandardAnalyzer()
    location = "/home/kevin/Downloads/Lab/test.index"
    store = SimpleFSDirectory(File(location))
    config = IndexWriterConfig(Version.LUCENE_4_10_1,analyzer)
    #config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(store, config)
    doc = Document()
    figid_field = StringField("figid", "PMC1312313/1471-2164-6-151-1", Field.Store.YES)
    doc.add(figid_field)
    writer.addDocument(doc)
    #writer.commit()
    #writer.optimize()
    writer.close()
    
def reader(queryString):
    lucene.initVM()
    vm_env = lucene.getVMEnv()
    vm_env.attachCurrentThread()
    analyzer = WhitespaceAnalyzer(Version.LUCENE_4_10_1)
    reader = IndexReader.open(SimpleFSDirectory(File(location)))
    searcher = IndexSearcher(reader)
    query = QueryParser(Version.LUCENE_4_10_1, "figid", analyzer).parse(queryString)
    
    MAX = 10000
    hits = searcher.search(query, MAX)
    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        print doc.get('figid')


'''
Methods for lucene 6.10
'''
def build_index(location):
    lucene.initVM()
    vm_env = lucene.getVMEnv()
    vm_env.attachCurrentThread()
    analyzer = StandardAnalyzer()
    location = "/home/kevin/Downloads/Lab/test.index"
    store = SimpleFSDirectory(File(location).toPath())
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(store, config)

    doc = Document()

    t1 = FieldType()
    #t1.indexOptions()
    t1.setStored(True)
    t1.setTokenized(False)
    t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
    
    doc.add(Field("figID", ("PMC1312313/1471-2164-6-151-1"), t1))
    writer.addDocument(doc)
    writer.commit()
    writer.close()
    store.close()

def read_index():
    lucene.initVM()
    vm_env = lucene.getVMEnv()
    vm_env.attachCurrentThread()
    analyzer = StandardAnalyzer()
    location = "/home/kevin/Downloads/Lab/test.index"
    store = SimpleFSDirectory(File(location).toPath())
    reader = DirectoryReader.open(store)
    searcher = IndexSearcher(reader)
    query = QueryParser("figID", analyzer)
    query.setAllowLeadingWildcard(True)
    query = query.parse(("PMC1312313\/1471-2164-6-151-1"))
    