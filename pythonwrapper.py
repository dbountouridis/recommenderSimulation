from __future__ import division
import numpy as np
from scipy import spatial
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

def recengine(engine,P,a,n,opdist):
	# %Engines
	# %1: CF , cosine similarity
	# %2: CF undiscounted in cosine similar , discounted in argmax afterward
	# %3: CF , TF-IDF before cosine similarty, non-discounted argmax afterward
	# %4: Designs 2+3
	# %5: Lowest selling
	# %6: Median selling
	# %7: Best selling
	# %8: Top 3 Sellers (note, this returns a vector)
	# %9: Random item
	# %11: CF , correlation 
	# %12: CF undiscounted in correlation, discounted in argmax afterward
	# %13: CF , TF-IDF before correlation, non-discounted argmax afterward
	# %P=User-Item purchase mx; a=active user num; n=num nearest neighbors

	if engine==1:
		norms = np.sqrt(np.sum(np.power(P,2),axis=1))
		norma = norms[a]
		sel = np.where(norms==0)[0]
		norms[sel] = 1

	if engine==5:		# lowest seller
		sales = np.sum(P, axis=0)
		Recommendation = np.argmin(sales)

	if engine==6:
		sales = np.sum(P, axis=0)
		v = np.argsort(sales).tolist()
		Recommendation = v[int(len(v)/2)]

	if engine==7:		# highest seller
		sales = np.sum(P, axis=0)
		Recommendation = np.argmax(sales)
		
	if engine==9:		# random
		Recommendation = int(random.random()*P.shape[1])

	return Recommendation


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

def LogitChoiceByHand(Distances, SimMetric, k, Spec, Delta, Rec, smoothhist, varBeta, w):

	if Spec==1: Distances[Rec] = Distances[Rec]*Delta

	#Convert distance to similarity based on metric
	if SimMetric == 1: Similarity = - Distances
	if SimMetric == 2: Similarity = - k*np.log(Distances)
	if SimMetric == 3: Similarity = np.power(Distances,-k)

	#Calc deterministic utility (based on utility spec desired)
	#spec: 0 = f(d), 1 = f(Delta*d), 2 = delta*f(d), 3 = f(d) + Delta
	V = 1*Similarity + varBeta*smoothhist

	#If spec==0, f(d)      +...  don't do anything: rec's off and no salience
	#If spec==1, f(Delta*d)+...  don't do anything: already multiplied above
	if Spec==2:  
		V[Rec] = Delta*1*Similarity[Rec] + varBeta*smoothhist[Rec]
	if Spec==3:
		V[Rec] = 1*Similarity[Rec] + Delta + varBeta*smoothhist[Rec]
	
	# utility
	R = [random.random() for v in range(len(V))]
	E = -np.log(-np.log(R))
	U = V + E
	sel = np.where(w==1)[0]
	mx = np.argmax(U[sel])
	i = sel[mx]
	return i # index of chosen item


def makeawaremx(awaremech,theta,Products,Dij,A,I,Lambda=0.75):
	awaremech==3   # not used currently
	Do = spatial.distance.cdist([[0,0]], Products)[0] 	# distance of products from the origin
	W = np.zeros([A,I])
	for a in range(A):
		for i in range(I):
			W[a,i] = Lambda*np.exp(-(np.power(Do[i],2))/(theta/1)) + (1-Lambda)*np.exp(-(np.power(Dij[a,i],2))/(theta/3))
			W[a,i] = random.random()<W[a,i] # probabilistic
	return W
	              

def sim2fcntimeseries(A,I,iters1,iters2,n,delta,IdealPoints,Products,engine,metric,k,plots,i,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda):
	if opdist<=0: # if not outer product
		P = np.zeros([A,I]) 	 # Purchases
		H = np.zeros([A,I]) 	 # Histories
		Pt = np.ones([I, iters1]) # For visualization
	else:
		P = np.zeros([A,I + 1]) # Purchases
		H = np.zeros([A,I + 1]) # Histories

	# Create distance matrix
	D = np.zeros([A,I])
	D = spatial.distance.cdist(IdealPoints, Products)
	Do = spatial.distance.cdist([[0,0]], Products)[0] 	# distance of points from origin


	# Make binary awareness matrix 
	W = makeawaremx(awaremech,theta,Products,D,A,I,Lambda)


	# Control period : recommendations off
	Ptemp = P.copy()
	selectedItems = []
	for t in range(iters1):
		if t==0:
			#Pt[indexOfChosenItem,t] = 1
			nothing = False
		else:
			Pt[:,t] = Pt[:,t-1]
		for a in range(A):
			indexOfChosenItem = LogitChoiceByHand(D[a,:], metric, k, 0, delta, 1, H[a,:], varBeta, W[a,:])
			
			Ptemp[a,indexOfChosenItem]+=1				# add to sales history
			selectedItems.append(indexOfChosenItem)
			H[a,:] = varAlpha*H[a,:]
			H[a,indexOfChosenItem] +=(1-varAlpha)	# update loyalty smooths
			
			Pt[indexOfChosenItem,t] +=1
	
	# Calculate Gini
	G = gini(np.sum(Ptemp,axis=0))
	print("Gini (control period):",G)

	# # Treament periof : recommendations on
	# Ptemp2 = Ptemp.copy() # either empty purchase list
	# # Ptemp2 = P[:]			 # or get the history from the control period
	# for t in range(iters2):
	# 	for a in range(A):
	# 		Rec = recengine(engine,Ptemp2,a,n,opdist) 	# recommends one item only
	# 		W[a,Rec] = 1 	# Rec forces permanent awareness !!!!
	# 		indexOfChosenItem = LogitChoiceByHand(D[a,:], metric, k, spec, delta, Rec, H[a,:], varBeta, W[a,:])
	# 		Ptemp2[a,indexOfChosenItem]+=1				# add to sales history
	# 		H[a,:] = varAlpha*H[a,:]
	# 		H[a,indexOfChosenItem] +=(1-varAlpha)	# update loyalty smooths

	# 		Pt[indexOfChosenItem,t+iters1] = Pt[indexOfChosenItem,t-1+iters1]+1
	
	# # since the recommendation period started with the purchase data of the control period do the following
	# Ptemp2 = Ptemp2 - Ptemp

	# G = gini(np.sum(Ptemp2,axis=0))
	# print("Gini (recommender period):",G)




	if plots:
		# product purchase accumulation. Central products are red, niche products are green. An ideal scenario will promote both
		colors = [ (1 - d/max(Do),  d/max(Do), 0,1 ) for d in Do]
		for i in range(Pt.shape[1]): Pt[:,i] = Pt[:, i]/np.sum(Pt[:, i])
		ind = np.argsort(Do)
		plt.stackplot(range(Pt.shape[1]),Pt, colors = colors)
		plt.show()


		# 2d visualization of products and users. Yellow circles show the overall amount of purchases in the end of the iterations
		n, bins, patches = plt.hist(selectedItems, bins=range(I+1),  facecolor='g', alpha=0.75)
		plt.close()
		ax = plt.gca()
		ax.scatter(IdealPoints[:,0], IdealPoints[:,1], marker='^', c='b',s=50)
		ax.scatter(Products[:,0], Products[:,1], marker='o', c='r',s=50)
		for i in range(I): plt.scatter(Products[i,0], Products[i,1], marker='o', c='y',s=400*(n[i]/np.max(n)),alpha=0.8)
		circle1 = plt.Circle((0, 0), 1, color='r',fill=False)
		ax.add_artist(circle1)
		plt.show()


## INPUTS
A = 50                         #Agents, users
I = 100                           #Items, products
dim = 2                          #Dimension of map
engine = 5                       #1=CF, 2=Inverse CF, 3=TFIDF-CF, 4=Lowest seller, 5=Median seller, 6=Highest seller, 7=Random, 8=TFIDF + Inv Weight
n = 5                            #Top-n similar used in collaborative filter

## Choice model
metric = 2                       #1=(-1)*Distance, 2= -k*Log(Distance), 3=(1/Distance),  2 from paper
# the higher the k the the consumer prefers closest products
k = 10                           #Constant used in similarity function,  10 from paper
opdist = 0                       #Outside good's distance 0=Off 0.75 in paper

## Variety seeking (Section 7.3)
varAlpha = 0.75                  #Variety: parameter governing exponential smooth, 0.75 in paper
# varBeta > 0 = inertia: consumers don't want to deviate, recommender
# influence limited
# varBeta < 0 = variety seeking:
varBeta = 0                    #Variety: coefficient of smooth history in utility function

## Salience
spec = 3                         #Utility spec for salience: 1=f(del*D), 2=del*f(D), 3=f(D)+del
# if delta is 0, then the item just becomes aware by the user. delta>0 increases prob of purchase/read
delta = 5                       #Factor by which distance decreases for recommended product

## Awareness, setting selectes predefined awareness settings
# paper uses setting = 2
# theta = 0.35
awaremech = 3                    #Awareness mech is wrt: 0=Off, 1=Origin,2=Person,3=Both (note, don't use 2 unless there exists outside good, will make fringe users aware of no products and cause 0 denominator for probability matrix)
theta = 0.35                        #Awareness Scaling, .35 in paper
Lambda = 0.75 	# This is crucial since it controls how much the users focus on mainstream items, 0.75 default value (more focused on mainstream)


## Iterations (for baseline iters1, and with recommenders on iters2)
iters1 = 100                      #Length of period without recommendations (all agents make 1 purchase/iteration)
iters2 = 100                      #Length of period with recommendations (uses sales data left at end of Iters1)
plots=1                       #1=produce all plots, 0=plots off

# experiment
IdealPoints = np.array([ [np.random.normal(), np.random.normal()] for i in range(A)])
Products = np.array([ [np.random.normal(), np.random.normal()] for i in range(I)])



sim2fcntimeseries(A,I,iters1,iters2,n,delta,IdealPoints,Products,engine,metric,k,plots,i,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda);
