import pickle
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.stats as stats

dic = pickle.load( open( "biodict.pickle", "rb" ) )
results =[]

dic1 = dic
G=nx.Graph()
dic3 = dict(dic)

for key in dic:
    if((re.sub("[ ]+", "", dic[key])!="") and len(dic[key]) ) > 30:
        G.add_node(key)
    else:
        del dic3[key]

dic1 = dic3

vect = TfidfVectorizer(min_df=1)    
coefs = list()

def embed_nodes(g, x, y):
    return dict((g.nodes()[i], (x[i],y[i])) for i in range(len(g.nodes())))

for key in dic3:
    for key1 in dic1:
        if(key!=key1):


            # print "Person: ", dic[key]
            # print "Person: ", dic1[key1]
            tfidf = vect.fit_transform([(dic[key]),(dic1[key1])])
            tfidf =(tfidf * tfidf.T).A[0,1]
            coefs.append(tfidf)
            #results.append(tfidf)
            prob = random.random()
            if (tfidf > 0.2   or prob < 0.001):
                G.add_edge(key,key1)

plt.hist(coefs)
print np.mean([c for c in coefs if c != 0])
print np.sum([c for c in coefs if c != 0])
print stats.mode([c for c in coefs if c != 0])
print np.max([c for c in coefs if c != 0])
print np.min([c for c in coefs if c != 0])
print np.std([c for c in coefs if c != 0])
plt.show()
outdeg = G.degree()
G.remove_nodes_from([n for n in outdeg if outdeg[n] == 0])

lapgr = nx.laplacian_matrix(G)
(w,v) = lin.eigh(lapgr.todense())
emb = embed_nodes(G,v[:,1].ravel().tolist()[0],v[:,2].ravel().tolist()[0])
# plt.figure(figsize=(,2))
# nx.draw_networkx(G, pos=emb, node_size=10, with_labels=False)

# nx.draw(G)


# j=3
# emb = dict((G.nodes()[i], (i, v[:,j].ravel().tolist()[0][i])) for i in range(len(G.nodes())))
# nx.draw_networkx(G, pos=emb, node_size=20, with_labels=False)         

# for j in range(20):
#     emb = dict((G.nodes()[i], (G.nodes()[i][0], j+v[:,j].ravel().tolist()[0][i])) for i in range(len(G.nodes())))
#     nx.draw_networkx(G, pos=emb, node_size=1, with_labels=False)  
plt.savefig("path_graphProb0.4.png")
plt.show() 

#print np.mean(results)   
         
