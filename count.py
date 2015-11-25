import pickle
dic = pickle.load( open( "biodict.pickle", "rb" ) )
i=0
for item in dic:
    i +=1
print i