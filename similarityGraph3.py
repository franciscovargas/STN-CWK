import pickle
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import eigs
import math
from numpy import linalg as LA

from sklearn.cluster import KMeans

def buildGraph():
    dic = pickle.load( open( "BiodictData/biodictdist4.pickle", "rb" ) )
    
    dic1 = dic
    G=nx.Graph()
    dic3 = dict(dic)

    
    for key in dic:
        if((re.sub("[ ]+", "", dic[key])!="") and len(dic[key]) ) > 30:
            G.add_node(key)
        else:
            del dic3[key]
    #print len(G.nodes())
    
    dic1 = dic3
    
    vect = TfidfVectorizer(min_df=1)    
    coefs = list()
    
    
    joint_dict = dict()
    dica = dict(dic3)
    dicb = dict(dic1)
    for key in dic3:
        for key1 in dic1:
            if(key!=key1):
                # print "Person: ", dic[key]
                # print "Person: ", dic1[key1]
                tfidf = vect.fit_transform([(dic[key]),(dic1[key1])])
                tfidf =(tfidf * tfidf.T).A[0,1]
                if dic[key]== dic[key1]:

                    continue
                else:
                    coefs.append(tfidf)
                #results.append(tfidf)
                    #prob = random.random()
                    joint_dict[str(key) + str(key1)] = tfidf
    
    
    # plt.hist(coefs)
    data = [c  for c in coefs if c != 0 ]
    #print max(data)
    
    mu = np.mean([c for c in coefs if c != 0])
    std = np.std([c for c in coefs if c != 0])
    #print mu
    #print std
    binwidth = 0.007
    #plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    #plt.show()

    
    for key in dic3:
        for key1 in dic1:
            if(key!=key1):
                try:
                    x = joint_dict[str(key) + str(key1)]
                    if x == 1.0:
                        print "goes in", x
                    prob = random.random()
                    if( x - mu > 2* std ):
                        G.add_edge(key,key1)
                except:
                    pass
                    
    G = max(nx.connected_component_subgraphs(G),key=len)
    return G

def laplacianMatrix(graph):   
    #graph = buildGraph()  
      
    A = nx.adjacency_matrix(graph)
    A = A.todense()
    degree = 0
    degreeVector = []
    #print (A)

    for i in range(0,(len(A))):

        for j in range(0, (A[i].size)):
            if(A[i,j]==1):
                degree += 1.0
                A[i,j] = -float(1.0)
            else:
                A[i,j] = float(0.0)
        A[i,i] = float(float(degree) - float(A[i,i]))    #Compute the Laplacian matrix inside of A
        if (degree!=0):
            degreeVector.append(float(1/np.sqrt(degree)))
        else:
            degreeVector.append(0)
        degree = 0
    
    identity = np.identity(len(A))
    degreeMatrix = identity*degreeVector
    #print degreeMatrix


    #print A 
    normA = degreeMatrix*A*degreeMatrix
    #print normA

    eigVal,eigVectors = eigs(normA, 2,  which='LM')
    #Val,Vec = LA.eig(A)
    return A, eigVectors
 

def splitGraph(graph, results):   
    graphChange1 = nx.Graph()
    graphChange2 = nx.Graph()
    i=0
    for node in graph:
        if(results[i]==0):
            graphChange1.add_node(node)
        else:
            graphChange2.add_node(node)
        i +=1
    #print graphChange2.nodes()
    flag=0
    flagd = 0
    for item in graph.edges():
        for node in graphChange1:
            if item[0] == node:
                flag += 1

    
        for node in graphChange1:
            if item[1] == node:
                flag += 1
        if flag == 2:
            graphChange1.add_edge(item[0],item[1])
        flag = 0
        for noded in graphChange2:
            if item[0] == noded:
                flagd += 1

    
        for noded in graphChange2:
            if item[1] == noded:
                flagd += 1
        if flagd == 2:
            graphChange2.add_edge(item[0],item[1])
        flagd = 0
    #print graphChange2.edges()
    #print graph.edges()
    return graphChange1, graphChange2
                   


def communityGraph(graph):
    
    lapgr = nx.laplacian_matrix(graph)
    print lapgr.T == lapgr

    evals,evec = np.linalg.eigh(lapgr.todense()) #Get the eigenvalues and eigenvectors of the Laplacian matrix
    print "Value", evals
    print "Vector", evec
    #print evals[2]==evals[3]
    #print list(graph.degree().values())

    fiedler = evec[1]
    results =[]
    #print "Fiedler", fiedler
    median = np.median(fiedler, axis=1) #median of the second eigenvalue
    for i in range (0,fiedler.size):    #divide the graph nodes into two
        if(fiedler[0,i]<median):
            results.append(0)
        else:
            results.append(1)
    return results, evals, evec
    
def embed_nodes(g, x, y):
    return dict((g.nodes()[i], (x[i],y[i])) for i in range(len(g.nodes())))  

def showEigenspace(G, V,k ):
    V = V.T[0:k,:]
    fig = plt.figure()
    #fif, axarr = plt.subplots(len(V), len(V))
    count =0 
    for i, v in enumerate(V):
        for j, u in enumerate(V):
            #print len(v)
            #print len(u)
            emb = embed_nodes(graph,v.ravel().tolist()[0],u.ravel().tolist()[0])
            plt.subplot(k,k, count)
            nx.draw_networkx_nodes(G, pos=emb, node_size=50, with_labels=False, node_color=G.degree().values(), cmap= plt.get_cmap('gray'))
            nx.draw_networkx_edges(G, pos=emb)
            # exit()
            count += 1
    plt.show()
    
def specKMeans(graph,evec, k):
    nodeCluster = dict()
    evec = evec.T[1:k+1,:]
    #print evec.shape
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(evec.T)
    centroids = kmeans.cluster_centers_
    print "Centroids", centroids
    i=0
    for node in graph:
        nodeCluster[node] = y_pred[i]
        i += 1
    return y_pred, nodeCluster

if __name__ == '__main__':
    graph = buildGraph()
    #print len(graph.nodes())

    #nx.draw_networkx(graph)
    #plt.show() 
    
    results, evals, evec = communityGraph(graph)
    kmeans, cluster = specKMeans(graph, evec, 2)
    #print cluster
    
    #first, second = splitGraph(graph,results)
    
    emb = embed_nodes(graph,evec[:,1].ravel().tolist()[0],evec[:,2].ravel().tolist()[0])
    
    showEigenspace(graph, evec, 5)
    '''
    nodelist1 = []
    nodelist2=[]
    i=0
    for key in graph:
        if(results[i]==0):
            nodelist1.append(key)
        else:
            nodelist2.append(key)
        i += 1
    #print "Nodelist", nodelist1
    #print "Second nodelist", nodelist2
    '''

    
    i=0
    for node in graph.nodes():
        graph.node[node]['category'] = kmeans[i]
        i += 1
    color_map = {0:'b', 1:'r', 2:'y', 3: 'g'}

    #nx.draw(graph, node_color=[color_map[graph.node[node]['category']] for node in graph]) 

    nx.draw_networkx(graph, pos=emb, node_size=100, with_labels=False, node_color=[color_map[graph.node[node]['category']] for node in graph])
    plt.show()
    '''
    nx.draw_networkx_nodes(graph,pos = emb,
                       nodelist=nodelist1,
                       node_color='r')
                      
    nx.draw_networkx_nodes(graph,pos=emb,
                       nodelist=nodelist2,
                       node_color='b')
    '''
    #nx.draw_networkx_edges(graph,pos=emb,width=1.0,alpha=0.5)
    
    
    
