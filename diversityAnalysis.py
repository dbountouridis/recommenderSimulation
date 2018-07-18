from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
from scipy.stats import norm
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import networkx as nx
import glob

def networkAnalysis(Data, type = 'preferences'):		
		AverageDistance = {}
		MedianDegree = {}
		Deviation = {}
		Skewness = {}
		print(Data.keys())

		# plot distributions
		fig, ax = plt.subplots(len(Data.keys()), sharex=True)
		L = list(Data.keys())
		L.pop(L.index("Control"))
		for i,eng in enumerate(["Control"]+L):

			# two types of analysis
			# on the 2d position (preferences, intent) of the users
			if type == "preferences":
				# convert distance matrix to similarity
				matrix = -(spatial.distance.cdist(Data[eng]["Users"], Data[eng]["Users"], metric = 'euclidean'))
				
			# on the users' purchase history
			if type == "purchases":
				matrix = 1-spatial.distance.cdist(Data[eng]["Sales History"], Data[eng]["Sales History"], metric = "cosine")

			if type == "profiles":
				# load item features
				ItemsDF = pd.read_pickle("temp/Items.pkl")
				df1 = ItemsDF.iloc[:,0:5]
				P = Data[eng]["Sales History"]
				print(P.shape)
				F = []
				for user in range(P.shape[0]):
					purchases = P[user,:]
					items = np.where(purchases==1)[0]
					userdf = df1.loc[df1.index.isin(items.tolist())]
					#print(userdf.mean())
					F.append(np.array(userdf.mean()))
					
					# fig, ax = plt.subplots(2, sharex=True)
					# ax[0].hist(items,normed=True,bins=P.shape[1])
					# plt.show()
					# for p in purchases:

				print(df1)
				matrix = 1-spatial.distance.cdist(F,F, metric = "cosine")

			
			ax[i].hist(matrix.flatten(), normed=True, bins=50)
		plt.show()

		for eng in ["Control"]+list(Data.keys()):

			# two types of analysis
			# on the 2d position (preferences, intent) of the users
			if type == "preferences":
				# convert distance matrix to similarity
				matrix = -(spatial.distance.cdist(Data[eng]["Users"], Data[eng]["Users"], metric = 'euclidean'))
				
			# on the users' purchase history
			if type == "purchases":
				matrix = 1-spatial.distance.cdist(Data[eng]["Sales History"], Data[eng]["Sales History"], metric = "cosine")

			if type == "profiles":
				# load item features
				ItemsDF = pd.read_pickle("temp/Items.pkl")
				df1 = ItemsDF.iloc[:,0:5]
				P = Data[eng]["Sales History"]
				print(P.shape)
				F = []
				for user in range(P.shape[0]):
					purchases = P[user,:]
					items = np.where(purchases==1)[0]
					userdf = df1.loc[df1.index.isin(items.tolist())]
					#print(userdf.mean())
					F.append(np.array(userdf.mean()))
					
					# fig, ax = plt.subplots(2, sharex=True)
					# ax[0].hist(items,normed=True,bins=P.shape[1])
					# plt.show()
					# for p in purchases:

				print(df1)
				matrix = 1-spatial.distance.cdist(F,F, metric = "cosine")

			# network 
			G = nx.Graph()

			# create nodes
			nodes = range(0,matrix.shape[0])
			for node in nodes: G.add_node(node)

			# add edges, at the same time store upper tringular matrix
			utmatrix = []
			for node1 in range(0,matrix.shape[0]):
				for node2 in range(node1+1,matrix.shape[0]):
					G.add_edge(node1, node2,weight=matrix[node1,node2])
					utmatrix.append(matrix[node1,node2])

			# Average distance
			if eng == "Control":
				AverageDistance.update({eng: np.mean(np.mean(utmatrix))})
			else:
				AverageDistance.update({eng: np.mean(np.mean(utmatrix))})

			# Average distance
			if eng == "Control":
				Deviation.update({eng: np.std(utmatrix)})
			else:
				Deviation.update({eng: np.std(utmatrix)})

			# Skewness
			if eng == "Control":
				Skewness.update({eng: stats.skew(utmatrix)})
			else:
				Skewness.update({eng:  stats.skew(utmatrix)})

			# Median degree
			S = []
			for node in nodes:
				S.append(np.sum([G[f][node]["weight"] for f in G.neighbors(node)]))
			if eng == "Control":
				MedianDegree.update({eng: np.median(S)})
			else:
				MedianDegree.update({eng:  np.median(S)})
			
			G.clear()
				
		return {"Average similarity":AverageDistance, "Median degree": MedianDegree, "Deviation": Deviation, "Skewness": Skewness }

# Gini coefficient computation
def gini(x):
	x = np.sort(x)
	n = x.shape[0]
	xbar = np.mean(x)
	#Calc Gini using unordered data (Damgaard & Weiner, Ecology 2000)
	absdif = 0
	for i in range(n):
		for j in range(n): absdif += abs(x[i]-x[j])
	G = absdif/(2*np.power(n,2)*xbar) * (n/(n)) # change to n/(n-1) for unbiased
	return G

# Read final output
files = glob.glob("temp//*-data.pkl")

Data = {}
engines = []
for i,file in enumerate(files):
	engine = file.split("temp/")[1].split("-data.pkl")[0]
	engines.append(engine)
	Data.update({engine: pickle.load(open(file, 'rb')) })

# # Gini coeff
# print("==== Gini coeff ====")
# for eng in Data.keys():
# 	# since the recommendation period started with the purchase data of the Control period do the following
# 	if eng!= "Control":
# 		D = Data[eng]["Sales History"] - Data["Control"]["Sales History"]
# 		G1 = gini(np.sum(Data["Control"]["Sales History"],axis=0))
# 		G2 = gini(np.sum(D,axis=0))
# 		print(eng,G2 - G1)

print("==== Network analysis ====")
print(networkAnalysis(Data, type="profiles"))
#print(networkAnalysis(Data, type="purchases"))
#print(networkAnalysis(Data, type="preferences"))




