from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2
import random
import time
import seaborn as sns
import pandas as pd
import pickle
import networkx as nx
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import os
import sys, getopt
import copy
import json
import metrics
import matplotlib

Us = 5000


# GMM on items/articles from the BBC data
R, S = [5,1,6,7], [5,2,28,28]
r = int(random.random()*4)
(X,labels,topicClasses) = pickle.load(open('BBC data/t-SNE-projection'+str(R[r])+'.pkl','rb'))
gmm = GaussianMixture(n_components=5, random_state=S[r]).fit(X)

# items 
samples_, classes_ = gmm.sample(1000)
ItemsClass = np.array(classes_)
ItemFeatures = gmm.predict_proba(samples_)
Items = samples_/55  # scale down to -1, 1 range

Users = np.random.uniform(-1,1,(Us,2))
for i, user in enumerate(Users):
	while spatial.distance.cdist([user], [[0,0]],metric = 'euclidean')[0][0]>1.1:
		user = np.random.uniform(-1,1,(1,2))[0]
	Users[i] = user
UsersClass = [gmm.predict([Users[i]*55])[0] for i in range(Us)]

D = spatial.distance.cdist(Users, Items, metric = 'euclidean')

Data = []
for T in range(10):
	k = 3
	for user in range(Us):
		Distances = D[user,:]
		Similarity = -k*np.log(Distances)  
		V = Similarity.copy()
		E = -np.log(-np.log([random.random() for v in range(len(V))]))
		U = V + E

		# with stochastic
		selected = np.argsort(U)[::-1]

		# without stochastic
		selectedW = np.argsort(V)[::-1]
		for i in selected[0:6]:
			 Data.append([ItemsClass[i],UsersClass[user],ItemsClass[i]==UsersClass[user],user])


df = pd.DataFrame(Data, columns = ["ItemClass", "UserClass","ItemTopic==UserMainTopic","User"])
#print(df)
f = []
for g,group in df.groupby("User"):
	t= np.sum(group["ItemTopic==UserMainTopic"])/np.array(group["ItemTopic==UserMainTopic"]).shape[0]
	f.append(t)
#print(np.mean(f))

f = []
for g,group in df.groupby("ItemClass"):
	#print(group)
	t= np.sum(group["ItemTopic==UserMainTopic"])/np.array(group).shape[0]
	print(t)
#print(np.mean(f))
	


