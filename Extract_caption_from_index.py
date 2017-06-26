#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:35:42 2017

@author: kevin
"""
import lucene
from lucene import *
from lucene import IndexWriter, StandardAnalyzer, Document, Field
from lucene import SimpleFSDirectory, File, initVM, Version
import re
BooleanQuery.setMaxClauseCount( Integer.MAX_VALUE )
location = "/home/kevin/mnt"
using_db = True

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
            #print doc.get('figid')
            #print doc.get('caption')
            fig_w_caption.append((doc.get('figid'), doc.get('caption')))
        import pickle
        with open("fig_w_caption.txt", "wb") as fp:   #Pickling
            pickle.dump(fig_w_caption, fp)


if __name__ == "__main__":
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
    
    import pickle
    with open("results_fig.txt", "rb") as fp:   # Unpickling
        content = pickle.load(fp)
    # you may also want to remove whitespace characters like `\n` at the end of each line
    fids = [str("figid:\"" + x.strip() + "\"") for x in content]
    fid_len = len(fids)-1
    query = ""
    for idx, fid in enumerate(fids):
        if idx < fid_len:
            query += fid + ' OR '
        else:
            query += fid
    
    reader(query)
    
#    q = session.query(fid_caption).all()
#    db_fids = []
#    for fid in q:
#        db_fids.append("figid:" + '"' + str(fid.fid) + '"')
#    not_in = len(set(fids)) - len(set(db_fids))