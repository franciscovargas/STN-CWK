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

# from aggregate_dist import agg_dist

def buildGraph(pickl):
    dic = pickl
    
    dic1 = dic
    G=nx.Graph()
    dic3 = dict(dic)
    checked = []
    
    for key in dic:
        if((re.sub("[ ]+", "", dic[key])!="") and len(dic[key]) ) > 30:
            # if(dic[key] not in checked ):
            G.add_node(key)
        else:
            del dic3[key]
    ## len(G.nodes())
    
    dic1 = dic3
    
    vect = TfidfVectorizer(min_df=1)    
    coefs = list()
    
    
    joint_dict = dict()
    dica = dict(dic3)
    dicb = dict(dic1)
    # Cosine similarity measure matrix
    F = vect.fit_transform(dic3.values())
    Cosine_mat = (F*F.T).A
    for i, key in enumerate(dic3):
        for j, key1 in enumerate(dic1):
            if(i > j):

                tfidf =Cosine_mat[i,j]
                if dic[key]== dic[key1]:

                    continue
                else:
                    coefs.append(tfidf)
                #results.append(tfidf)
                    #prob = random.random()
                    joint_dict[str(key) + str(key1)] = tfidf
    
    
    # plt.hist(coefs)
    # plt.show()
    data = [c  for c in coefs if c != 0 ]
    ## max(data)
    
    mu = np.mean([c for c in coefs if c != 0])
    std = np.std([c for c in coefs if c != 0])
    ## mu
    ## std
    binwidth = 0.007
    plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    plt.show()

    
    for key in dic3:
        for key1 in dic1:
            if(key!=key1):
                try:
                    x = joint_dict[str(key) + str(key1)]
                    if x == 1.0:
                        # "goes in", x
                        pass
                    prob = random.random()
                    if( x - mu > 2* std ):
                        G.add_edge(key,key1)
                except:
                    pass

    # Return the conected component with largest cardinality of nodes 
    G = max(nx.connected_component_subgraphs(G),key=len)
    return G

def laplacianMatrix(graph):   
    #graph = buildGraph()  
      
    A = nx.adjacency_matrix(graph)
    A = A.todense()
    degree = 0
    degreeVector = []
    ## (A)

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
    ## degreeMatrix


    ## A 
    normA = degreeMatrix*A*degreeMatrix
    ## normA

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
    ## graphChange2.nodes()
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
    ## graphChange2.edges()
    ## graph.edges()
    return graphChange1, graphChange2
                   


def communityGraph(graph):
    """
    Median method of community detection
    """
    
    lapgr = nx.laplacian_matrix(graph)
    # lapgr.T == lapgr

    evals,evec = np.linalg.eigh(lapgr.todense()) #Get the eigenvalues and eigenvectors of the Laplacian matrix
    # "Value", evals
    # "Vector", evec
    ## evals[2]==evals[3]
    ## list(graph.degree().values())

    fiedler = evec[1]
    results =[]
    ## "Fiedler", fiedler
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
    """
    This method projects our graph G in to a sub
    eigen living in R^k , Mostly used for visualization
    purposes
    """
    V = V.T[0:k,:]
    fig = plt.figure()
    #fif, axarr = plt.subplots(len(V), len(V))
    count =0 
    for i, v in enumerate(V):
        for j, u in enumerate(V):
            ## len(v)
            ## len(u)
            emb = embed_nodes(G,v.ravel().tolist()[0],u.ravel().tolist()[0])
            plt.subplot(k,k, count)
            nx.draw_networkx_nodes(G, pos=emb, node_size=50, with_labels=False, node_color=G.degree().values(), cmap= plt.get_cmap('gray'))
            nx.draw_networkx_edges(G, pos=emb)
            # exit()
            count += 1
    plt.show()
    
def specKMeans(graph,evec, k):
    nodeCluster = dict()
    evec = evec.T[1:k+1,:]
    ## evec.shape
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(evec.T)
    centroids = kmeans.cluster_centers_
    # "Centroids", centroids
    i=0
    for node in graph:
        nodeCluster[node] = y_pred[i]
        i += 1
    return y_pred, nodeCluster, centroids

def construct(pickl,startVector, endVector):
    graph = buildGraph(pickl)

        
    results, evals, evec = communityGraph(graph)
    kmeans, cluster, centroids = specKMeans(graph, evec, 2)
    
    #first, second = splitGraph(graph,results)
    franVectors = list()
    for i in range(0,len(evec[:,startVector])):
        
        franVectors.append([evec[:,startVector].item(i),evec[:,endVector].item(i)])
    
    
    emb = embed_nodes(graph,evec[:,startVector].ravel().tolist()[0],evec[:,endVector].ravel().tolist()[0])
    
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
    ## "Nodelist", nodelist1
    ## "Second nodelist", nodelist2
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
    return centroids, franVectors
def fran(startVector, endVector):
    pickl = pickle.load( open( "BiodictData/biodictdist4.pickle", "rb" ) )
    centr1, franVectors1 = construct(pickl, startVector, endVector)
    pickl = pickle.load( open( "BiodictData/biodictdist8.pickle", "rb" ) )
    centr2, franVectors2 = construct(pickl, startVector, endVector)
    centr = list()
    franVectors = list()
    centr.append(centr1)
    centr.append(centr2)
    franVectors.append(franVectors1)
    franVectors.append(franVectors2)
    return centr, franVectors
    

if __name__ == '__main__':
    k=2
    '''
    pickl = pickle.load( open( "biodictdist4.pickle", "rb" ) )
    centr1 = construct(pickl)
    pickl = pickle.load( open( "biodictdist8.pickle", "rb" ) )
    centr2 = construct(pickl)
    '''
    centr, franVectors = fran(1,2) 
    # # "Centroids" , centr
    # # "Vectors", franVectors
    # "time"
    # agg_dist(franVectors, centr)
    '''
    centroids = list()
    dist = list()
    for i in xrange(k):
        for j in xrange(k):
            dist.append( np.linalg.norm(centr1[j]-centr2[j]))
        centroids.append( min(dist))
    # centroids
    '''
    
    
    
