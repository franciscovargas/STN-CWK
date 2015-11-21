import pickle
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

dic = pickle.load( open( "biodict.pickle", "rb" ) )
results =[]

dic1 = dic
G=nx.Graph()
for key in dic:

    G.add_node(key)
vect = TfidfVectorizer(min_df=1)    


for key in dic:
    for key1 in dic1:
        if(key!=key1):

            if(dic[key] and dic1[key1]):

                tfidf = vect.fit_transform([(dic[key]),(dic1[key1])])
                tfidf =(tfidf * tfidf.T).A[0,1]
                #results.append(tfidf)
                prob = random.random()
                if (tfidf >0.15 and prob<0.4):
                    G.add_edge(key,key1)
            else:
                if (random.random()<0.1):
                    G.add_edge(key,key1)
            #print tfidf
            
nx.draw(G)  
plt.savefig("path_graphProb0.4.png")
plt.show() 

#print np.mean(results)   
         
