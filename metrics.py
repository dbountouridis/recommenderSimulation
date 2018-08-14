from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
from scipy.stats import norm
from sklearn import metrics
import random
import time


''' 

	Long Tail Novelty, as defined, is concerned with providing less popular, obvious
	recommendations. Under this perspective, an item is novel if few people are aware
	it exist, i. e. the item is far in the long tail of the popularity distribution.

	 interested in avoiding recommending a highly reduced set of the most popular items, the socalled
	short head, and promoting instead recommendations in the more numerous,
	less popular long tail. The popularity of an item is defined by how many
	users know about it, and an approximation to such information is readily available
	in most recommendation scenarios as the user-item interaction data in the
	form of a rating or play count matrix
'''

# Expected Popularity Complement (EPC)  
def EPC(Rec,RecAsMatrix,M,U_,Rtest):
	# Cu = 1/ np.sum(np.sum(RecAsMatrix))# temp
	A = []
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			sum_+= (1 - np.sum(Ui(item,M))/U_)*disc(i)*Prel(item,u,Rtest)
		A.append(sum_*Cu)
	print("EPC:",np.mean(A))
	return (np.mean(A),np.std(A))

# Expected Free Discovery (EFD)
# 4.3.2.1 Discovery-Based Measurement
def EFD(Rec,RecAsMatrix,M,U_,Rtest):
	A = []
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			top = np.sum(Ui(item,M))
			bottom = np.sum(np.sum(M))
			sum_+= np.log2(top/bottom)*disc(i)*Prel(item,u,Rtest)
		A.append(sum_*(-Cu))
	print("EFD:",np.mean(A))
	return np.mean(A),np.std(A)
			
''' 
	A related but different notion considers the Unexpectedness (Murakami et al., 2008;
	Zhang et al., 2012; Adamopoulos and Tuzhilin, in press) involved in receiving recommendations
	that are novel in the sense that they are different or unfamiliar to
	the user experience. 
'''
# Expected Profile Distance (EPD)
def EPD(Rec,RecAsMatrix,M,U_,Rtest,dist):
	A = [] 
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		Cu_ = np.sum([Prel(i,u,Rtest) for i in np.where(Iu(u, M)>=1)[0]])
		Iuu  = np.where(Iu(u, M)>=1)[0]
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			for itemj in Iuu:
				sum_ += dist[item,itemj]*disc(i)*Prel(item,u,Rtest)*Prel(itemj,u,Rtest)
		A.append((Cu/Cu_)*sum_)
		#print(Cu,Cu_,np.where(Iu(u, M)>=1)[0])
		#time.sleep(1)
	print("EPD",np.mean(A))
	return np.mean(A),np.std(A)

# Expected Intra-List Distance (EILD)
def EILD(Rec,RecAsMatrix,M,U_,Rtest,dist):
	A = [] 
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			Ci = Cu/np.sum([disc(max(0,j-i))*Prel(itemj,u,Rtest) for j,itemj in enumerate(Rec[u])])
			for j,itemj in enumerate(Rec[u]):
				if j>i:
					sum_ += dist[item,itemj]*disc(i)*disc(max(0,j-i))*Prel(item,u,Rtest)*Prel(itemj,u,Rtest)*Ci
		A.append(sum_)
		#print(Cu,Cu_,np.where(Iu(u, M)>=1)[0])
		#time.sleep(1)
	print("EILD",np.mean(A))
	return np.mean(A), np.std(A)


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
	return (1/(R_*(R_-1)))*sum_, 0





# user and item interaction profiles
def Iu(u, M):
	return M[u,:]

def Ui(i, M):
	return M[:,i]

def Prel(i, u, Mr):
	if Mr[u,i]>=1: return 1
	else: return 0.01

# user rec profile
def R(u,R):
	return R[u,:]

# simple exponential discount
#disc(kj| ki) = disc(max(0, kj −ki)) 
def disc(k):
	beta = 0.9
	return np.power(beta, k)


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



def metrics(M,Rec,ItemFeatures,dist,Mafter):
	U_ = M.shape[0]
	I_ = M.shape[1]
	Rtest = Mafter - M
	RecAsMatrix = np.zeros((U_, I_))
	for u in Rec.keys():
		RecAsMatrix[u,Rec[u]]=1
	(mEPC,sEPC) = EPC(Rec,RecAsMatrix,M,U_,Rtest)
	(mILD,sILD) = ILD(Rec,RecAsMatrix,M,U_,dist)
	(mEFD,sEFD) = EFD(Rec,RecAsMatrix,M,U_,Rtest)
	(mEPD,sEPD) = EPD(Rec,RecAsMatrix,M,U_,Rtest,dist)
	(mEILD,sEILD) = EILD(Rec,RecAsMatrix,M,U_,Rtest,dist)
	return {"EPC" :  mEPC,
	"EPCstd" :  sEPC,
	"ILD": mILD,
	"ILDstd": sILD,
	"EFD": mEFD,
	"EFDstd": sEFD,  
	"EPD": mEPD,
	"EPDstd": mEPD,
	"EILD": mEILD,
	"EILDstd": sEILD}
	
	# return {"EPC" : EPC(Rec,RecAsMatrix,M,U_,Rtest), 
	# "ILD": ILD(Rec,RecAsMatrix,M,U_,dist), 
	# "EFD": EFD(Rec,RecAsMatrix,M,U_,Rtest), 
	# "EPD": EPD(Rec,RecAsMatrix,M,U_,Rtest,dist),
	# "EILD": EILD(Rec,RecAsMatrix,M,U_,Rtest,dist)}
	# EPC(Rec,RecAsMatrix,M,U_)
	# EFD(Rec,RecAsMatrix,M,U_)
	# ILD(Rec,RecAsMatrix,M,U_,dist)
	#EPD(Rec,RecAsMatrix,M,U_,dist)
	