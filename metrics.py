from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
from scipy.stats import norm
from sklearn import metrics
import random
import time


# # Expected Free Discovery (EFD)
# def EFD(Rec,RecAsMatrix,M,U_):
# 	A = []
# 	for u in range(U_):
# 		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
# 		sum_ = 0
# 		for i,item in enumerate(Rec[u]):
# 			top = np.sum(Ui(item,M))
# 			bottom = np.sum(np.sum(M))
# 			sum_+= np.log2(top/bottom)*disc(i) 
# 		A.append(sum_*(-Cu))
# 	#print("EFD:",np.mean(A))

# # Expected Profile Distance (EPD)
# def EPD(Rec,RecAsMatrix,M,U_,dist):
# 	A = []
# 	for u in range(U_):
# 		#Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
# 		sum_ = 0
# 		Iuu = np.where(Iu(u, M)>=1)[0]
# 		#print(Iuu)
# 		for item in Rec[u]:
# 			for itemj in Iuu:
# 				sum_ += dist[item,itemj]
# 		#print(sum_,np.sum(Iu(u, M)))
# 		A.append(sum_/(np.sum(Iu(u, M)) ))
# 	#print("EPD:",np.mean(A),A)
# 	return np.mean(A)


# user and item interaction profiles
def Iu(u, M):
	return M[u,:]

def Ui(i, M):
	return M[:,i]

def Prel(i, u, Mr):
	if Mr[u,i]>=1: return 1
	else: return 0

# user rec profile
def R(u,R):
	return R[u,:]

# simple exponential discount
def disc(k):
	beta = 0.9
	return np.power(beta, k-1)


'''' 
	Metrics

'''
# Diversity measure: gini coefficients
# based on: Kartik Hosanagar, Daniel Fleder (2008)
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

def computeGinis(S, C):
	GiniPerRec = {}
	
	# since the recommendation period started with the purchase data of the control period do the following
	S = S - C
	G1 = gini(np.sum(C,axis=0))
	G2 = gini(np.sum(S,axis=0))
	return G2 - G1

# Expected Popularity Complement (EPC) 
def EPC(Rec,RecAsMatrix,M,U_,Rtest):
	# Cu = 1/ np.sum(np.sum(RecAsMatrix))# temp
	A = []
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			sum_+= (1 - np.sum(Ui(item,M))/U_)*disc(i)#*Prel(item,u,Rtest)
		A.append(sum_*Cu)
	#print("EPC:",np.mean(A))
	return(np.mean(A))
			
# Intra-List Distance
def ILD(Rec,RecAsMatrix,M,U_,dist):
	allR = np.where(np.sum(RecAsMatrix,axis=0)>=1)[0]
	#print(allR)
	sum_ = 0
	for item in allR:
		for itemj in allR:
			sum_ += dist[item,itemj]
	R_ = np.sum(np.sum(RecAsMatrix))
	#print("ILD:",1/(R_*(R_-1))*sum_ )
	return (1/(R_*(R_-1)))*sum_ 


def metrics(M,Rec,ItemFeatures,dist,Mafter):
	U_ = M.shape[0]
	I_ = M.shape[1]
	Rtest = Mafter - M
	RecAsMatrix = np.zeros((U_, I_))
	for u in Rec.keys():
		RecAsMatrix[u,Rec[u]]=1

	return {"EPC" : EPC(Rec,RecAsMatrix,M,U_,Rtest), "ILD"
	:ILD(Rec,RecAsMatrix,M,U_,dist) }
	# EPC(Rec,RecAsMatrix,M,U_)
	# EFD(Rec,RecAsMatrix,M,U_)
	# ILD(Rec,RecAsMatrix,M,U_,dist)
	#EPD(Rec,RecAsMatrix,M,U_,dist)
	