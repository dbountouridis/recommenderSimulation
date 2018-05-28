from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from scipy.stats import kendalltau


__author__ = 'Dimitrios Bountouridis'


# Recommendation algorithms (engines)
def recengine(engine,P,a,n,opdist):
	if engine=="CF": # standard cosine
		# Build row vector s := cosine(a, all users)
		# If a's sales are all 0, similarities are all 0
    	# If another user's sales are all 0, similarity to a is 0
		
		norms = np.sqrt(np.sum(np.power(P,2),axis=1))
		norma = norms[a]
		sel = np.where(norms==0)[0]
		norms[sel] = 1
		k = np.power(np.diag(norms),-1)
		kinf = np.where(np.isinf(k))
		k[kinf] = 0
		if norma>0: norms = k*(1.0/norma)
		else: norms = k
		s = np.matmul(norms,np.matmul(P,P[a,:]))
		s[a] = 0 # self-similarity set to 0
		if np.sum(s) == 0: # if no one similar, suggest most popular product
			Recommendation = np.argmax(np.sum(P,axis=0))
		else:
			topN = np.argsort(s)[::-1][:n] # top N users
			# Recommend the most common item among those n users
			SubP = P[topN,:]
			Recommendation = np.argsort(np.sum(SubP,axis=0))[::-1][0]

	if engine=="CFnorm": #CF with discounted argmax i.e. normalized by the sum of the product sales
		norms = np.sqrt(np.sum(np.power(P,2),axis=1))
		norma = norms[a]
		sel = np.where(norms==0)[0]
		norms[sel] = 1
		k = np.power(np.diag(norms),-1)
		kinf = np.where(np.isinf(k))
		k[kinf] = 0
		if norma>0: norms = k*(1.0/norma)
		else: norms = k
		s = np.matmul(norms,np.matmul(P,P[a,:]))
		s[a] = 0 # self-similarity set to 0
		if np.sum(s) == 0: # if no one similar, suggest most popular product
			Recommendation = np.argmax(np.sum(P,axis=0))
		else:
			topN = np.argsort(s)[::-1][:n] # top N users
			# Recommend the most common item among those n users
			SubP = P[topN,:]

			allSales = np.sum(P,axis=0)
			allSales[np.where(allSales==0)[0]]=1 # diving by one won't change results
			Recommendation = np.argsort(np.sum(SubP,axis=0)/allSales)[::-1][0]

		
	if engine=="min":		# lowest seller
		sales = np.sum(P, axis=0)
		Recommendation = np.argmin(sales)

	if engine=="median":		# median seller
		sales = np.sum(P, axis=0)
		v = np.argsort(sales).tolist()
		Recommendation = v[int(len(v)/2)]

	if engine=="max":		# highest seller
		sales = np.sum(P, axis=0)
		Recommendation = np.argmax(sales)

	if engine=="top5":		# Top-5 sellers
		sales = np.sum(P, axis=0)
		Recommendation = np.argsort(sales)[::-1][:5]
		
	if engine=="random":		# random
		Recommendation = int(random.random()*P.shape[1])

	return Recommendation


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


# Probabilistic function that selects a product for a user
def LogitChoiceByHand(Distances, SimMetric, k, Spec, Delta, Rec, smoothhist, varBeta, w):

	if Spec==1: Distances[Rec] = Distances[Rec]*Delta

	# Convert distance to similarity based on metric
	if SimMetric == 1: Similarity = - Distances
	if SimMetric == 2: Similarity = - k*np.log(Distances)
	if SimMetric == 3: Similarity = np.power(Distances,-k)

	# Calc deterministic utility (based on utility spec desired)
	#spec: 0 = f(d), 1 = f(Delta*d), 2 = delta*f(d), 3 = f(d) + Delta
	V = 1*Similarity + varBeta*smoothhist

	# If spec==0, f(d)      +...  don't do anything: rec's off and no salience
	# If spec==1, f(Delta*d)+...  don't do anything: already multiplied above
	if Spec==2:  
		V[Rec] = Delta*1*Similarity[Rec] + varBeta*smoothhist[Rec]
	if Spec==3:
		V[Rec] = 1*Similarity[Rec] + Delta + varBeta*smoothhist[Rec]

	# utility
	R = np.array([random.random() for v in range(len(V))])
	#print(R, R.shape)
	#time.sleep(1)
	E = -np.log(-np.log(R))
	#print(E,E.shape)
	# time.sleep(10)
	U = V + E*10
	#print(V,V.shape)
	sel = np.where(w==1)[0]
	mx = np.argmax(U[sel])
	i = sel[mx]
	#print(sel,U[sel],np.argmax(U[sel]),sel[mx])
	#time.sleep(1)
	return i, U # index of chosen item


# Create initial awareness matrix
def makeawaremx(awaremech,theta,Products,Dij,A,I,Lambda=0.75):
	awaremech==3   # not used currently
	Do = spatial.distance.cdist([[0,0]], Products)[0] 	# distance of products from the origin
	W = np.zeros([A,I])
	for a in range(A):
		for i in range(I):
			W[a,i] = Lambda*np.exp(-(np.power(Do[i],2))/(theta/1)) + (1-Lambda)*np.exp(-(np.power(Dij[a,i],2))/(theta/3))
			W[a,i] = random.random()<W[a,i] # probabilistic
	return W
	 

# Plotting analytics	    
def plotAnalysis(Data,varBeta, Products, Users):
	sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})

	# binsI = stats.mstats.mquantiles(Do, [i/20 for i in range(20)]+[1])
	# index = [np.argmin(abs(binsI-i))  for i in Do]
	#colors = [ (0, 1, 0 , d/max(Do)) for d in binsI]
	# f, ax = plt.subplots(1+len(engine), sharex=False, figsize=(12,8))
	# for p, period in enumerate(["Control"]+engine):
	# 	# product purchase accumulation. Central products are red, niche products are green. An ideal scenario will promote both
		
	# 	C = Data[period]["Product Sales Time Series"].copy()
	# 	for i in range(Data[period]["Product Sales Time Series"].shape[1]): 
	# 		d_ = Data[period]["Product Sales Time Series"][:, i]/np.sum(Data[period]["Product Sales Time Series"][:, i])
	# 		C[index,i]=d_
	# 	ax[p].stackplot(range(Data[period]["Product Sales Time Series"].shape[1]),C)#,colors = colors)
	# 	ax[p].set_ylim([0,1])
	# 	if p==0: ax[p].set_xlim([0,iters1])
	# 	else: ax[p].set_xlim([0,iters2])
	# 	ax[p].set_ylabel(period)
	# plt.show()
	# plt.close()

	# plot users, products on 2d plane
	f, ax = plt.subplots(1,1+len(engine), figsize=(15,6), sharey=True)
	for p, period in enumerate(["Control"]+engine):
		n, bins = np.histogram(Data[period]["All Purchased Products"], bins=range(I+1))
		circle1 = plt.Circle((0, 0), 1, color='g',fill=True,alpha=0.3,zorder=-1)
		ax[p].add_artist(circle1)
		#ax[p].scatter(Data[period]["InitialUsers"][:,0], Data[period]["InitialUsers"][:,1], marker='.', c='b',s=40,alpha=0.3)
		for i in range(len(Users[:,1])):
			ax[p].scatter(Data[period]["Users"][i,0], Data[period]["Users"][i,1], marker='.', c='b',s=40, alpha = 0.6 )
		for i in range(len(Users[:,1])):
			ax[p].plot([Data[period]["InitialUsers"][i,0], Data[period]["Users"][i,0]], [Data[period]["InitialUsers"][i,1], Data[period]["Users"][i,1]], 'b-', lw=0.5, alpha =0.6)
		#ax[p].scatter(InitUsers[:,0], InitUsers[:,1], marker='.', c='y',s=40,alpha=0.6)
		for i in range(I): 
			if n[i]>=1:
				v = 0.6# 0.4+ n[i]/np.max(n)*0.4
				c = (1,0,0.0,v)
				s = 2+n[i]/np.max(n)*40
				marker='o'
			else:
				c = (0,0,0,0.8)
				s = 10
				marker='x'
			ax[p].scatter(Products[i,0], Products[i,1], marker=marker, c=c,s=s)		
		ax[p].set_xlabel(period)
		ax[p].set_aspect('equal', adjustable='box')
	plt.tight_layout()
	plt.savefig("plots/users-products.pdf")
	plt.show()

	# plot only the users
	f, ax = plt.subplots(1,1+len(engine), figsize=(15,6), sharey=True)
	for p, period in enumerate(["Control"]+engine):
		n, bins = np.histogram(Data[period]["All Purchased Products"], bins=range(I+1))
		circle1 = plt.Circle((0, 0), 1, color='g',fill=True,alpha=0.3,zorder=-1)
		ax[p].add_artist(circle1)
		#ax[p].scatter(Data[period]["InitialUsers"][:,0], Data[period]["InitialUsers"][:,1], marker='.', c='b',s=40,alpha=0.3)
		for i in range(len(Users[:,1])):
			ax[p].scatter(Data[period]["Users"][i,0], Data[period]["Users"][i,1], marker='.', c='b',s=40,alpha=0.2+0.8*((varBeta[i]+20)/40) )
		ax[p].set_xlabel(period)
		ax[p].set_aspect('equal', adjustable='box')
	plt.tight_layout()
	plt.savefig("plots/users.pdf")
	plt.show()

	# histogram of user distances (to investigate their clusterdness)
	f, ax = plt.subplots(1,1+len(engine), figsize=(15,6), sharey=True)
	for p, period in enumerate(["Control"]+engine):
		x = spatial.distance.cdist(Data[period]["Users"], Data[period]["Users"])
		sns.distplot(x.flatten(),ax=ax[p])
		ax[p].set_xlabel(period)
	plt.savefig("plots/users-dist-distribution.pdf")
	plt.show()

def plots2(Products,Users,W):
	
	
	sns.set(style="ticks")
	indeces= np.array(W[0])==1
	fig, ax = plt.subplots()
	g = sns.jointplot(Products[indeces,0], Products[indeces,1], ax=ax,kind="kde",  stat_func=kendalltau, color="#4CB391",xlim=(-3,3),ylim=(-3,3))#,joint_kws=dict(shade_lowest=False))
	circle1 = plt.Circle((0, 0), 1, color='k',fill=False,alpha=0.5,zorder=1, edgecolor='r', lw=3)
	ax.add_artist(circle1)
	for i in range(len(Users[:,1])):
		ax.scatter(Users[i,0], Users[i,1], marker='.', c='b',s=40, alpha = 0.6 )
		
	ax.set_ylim(-3,3)
	ax.set_xlim(-3,3)
	ax.set_aspect('equal', adjustable='box')
	plt.show()

def plots3(Products,Users,selectedItems):
	sns.set(style="ticks")

	#selectedItems = list(set(selectedItems))

	P = []
	for s in selectedItems: 
		jitterx = 0 # random.random()*0.2-0.1
		jittery = 0# random.random()*0.2-0.1
		P.append([Products[s,0]+jitterx,Products[s,1]+jittery])
	P = np.array(P)
	
	fig, ax = plt.subplots()
	g = sns.jointplot(P[:,0], P[:,1], ax=ax,kind="kde",  stat_func=kendalltau, color="#4CB391",xlim=(-3,3),ylim=(-3,3))#,joint_kws=dict(shade_lowest=False))
	ax.scatter(P[:,0], P[:,1], marker='.', c='r',s=60, alpha = 0.01 )
	
	circle1 = plt.Circle((0, 0), 1, color='k',fill=False,alpha=0.5,zorder=1, edgecolor='r', lw=3)
	ax.add_artist(circle1)
	for i in range(len(Users[:,1])):
		ax.scatter(Users[i,0], Users[i,1], marker='.', c='b',s=40, alpha = 0.6 )
		
	ax.set_ylim(-2,2)
	ax.set_xlim(-2,2)
	ax.set_aspect('equal', adjustable='box')
	plt.show()

def plots4(Products,Users,U):
	sns.set(style="ticks")

	#selectedItems = list(set(selectedItems))
	U = U - np.min(U)
	U = U/np.max(U)

	fig, ax = plt.subplots()
	for i in range(len(Products[:,0])):
		ax.scatter(Products[i,0], Products[i,1], marker='.', c='r',s=60, alpha = U[i] )
	#g = sns.jointplot(P[:,0], P[:,1], ax=ax,kind="kde",  stat_func=kendalltau, color="#4CB391",xlim=(-3,3),ylim=(-3,3))#,joint_kws=dict(shade_lowest=False))
	circle1 = plt.Circle((0, 0), 1, color='k',fill=False,alpha=0.4,zorder=1, edgecolor='r', lw=3)
	ax.text(1+0.3, 1, "Mainstream", ha="center", va="center", size=12)
	ax.add_artist(circle1)
	for i in range(len(Users[:,1])):
		ax.scatter(Users[i,0], Users[i,1], marker='.', c='b',s=200, alpha = 0.6 )
		ax.text(Users[i,0]+0.6, Users[i,1], "User", ha="center", va="center", size=12)
		
	ax.set_ylim(-2,2)
	ax.set_xlim(-2,2)
	ax.set_aspect('equal', adjustable='box')
	plt.show()

# Main simulation function
def sim2fcntimeseries(A,I,iters1,iters2,n,delta,Users,Products,engine,metric,k,plots,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda,timer, percentageOfActiveUsers, percentageOfActiveItems, added, moveAsDistancePercentage):
	
	# Initialize stuctures
	if opdist<=0: 	# if not outer product
		P = np.zeros([A,I]) 	 # Purchases, sales history
		H = np.zeros([A,I]) 	 # Loyalty histories
		InitUsers = Users.copy()

		# Create distance matrices
		D = np.zeros([A,I])
		D = spatial.distance.cdist(Users, Products)					# distance of products from users
		Do = spatial.distance.cdist([[0,0]], Products)[0] 			# distance of products from origin, remains fixed for each engine

		# Make binary awareness matrix 
		W = makeawaremx(awaremech,theta,Products,D,A,I,Lambda)
		for i in range(0,50):
			W += makeawaremx(awaremech,theta,Products,D,A,I,Lambda)
		#plots4(Products,Users,W[0,:])
		
		# Make timer matrix for awareness
		T = W.copy()
		indecesOfInitialAwareness = W==1
		T[indecesOfInitialAwareness] = timeValue	

		# Make a dictionary structure
		Data = {}
		for eng in engine+["Control"]:
			if eng=="Control": iters = iters1
			else: iters = iters2
			Data.update({eng:{"Product Sales Time Series": np.ones([I, iters]), "Sales History": P.copy(),"All Purchased Products":[],"Users":Users.copy(),"InitialUsers":Users.copy(),"Awareness":W.copy(),"D":D.copy(),"T":T.copy(),"H":H.copy(),"Iterations":iters}})

	
	# Simulation per engine, starting with the Control
	selectedItems = np.zeros([A,I])
	UtilityPlot = np.zeros([A,I])
	for dd in range(0,500):
		print(dd)
	
		activeUserIndeces = np.arange(A).tolist()
		activeItemIndeces = np.arange(I).tolist()
	
		W = makeawaremx(awaremech,theta,Products,D,A,I,Lambda)
		W__ = W.copy()
		indexOfChosenItem, U =  LogitChoiceByHand(Data[eng]["D"][0,:], metric, k, spec, delta, 0, Data[eng]["H"][0,:], varBeta[0], W__[0,:])
		
		sel = np.where(W__[0,:]==1)[0]
		U = (U -np.min(U))
		U = U/np.sum(U)
		UtilityPlot[0,sel]=+U[sel]
		
		selectedItems[0,indexOfChosenItem]+=1
		
		
	plots4(Products,Users,UtilityPlot[0,:])
	plots4(Products,Users,selectedItems[0,:])
			
			
		


# Inputs
A = 1                         # Agents, users
I = 2000                         # Items, products
engine = ["CF","CFnorm","min","random"]#,"max"]#,"random","median"]                      
n = 5                           # Top-n similar used in collaborative filter

# Choice model
metric = 2                      # 1=(-1)*Distance, 2= -k*Log(Distance), 3=(1/Distance),  2 from paper
# the higher the k the the consumer prefers closest products
k = 10                          # Constant used in similarity function,  10 from paper
opdist = 0                      # Outside good's distance 0=Off 0.75 in paper

# Variety seeking 
varAlpha = 0.75                 # Variety: parameter governing exponential smooth, 0.75 in paper
# varBeta > 0 = inertia: consumers don't want to deviate, recommender
# influence limited
# varBeta < 0 = variety seeking:
varBeta = 0                     # Variety: coefficient of smooth history in utility function

# Salience
spec = 3                        # Utility spec for salience: 1=f(del*D), 2=del*f(D), 3=f(D)+del
# if delta is 0, then the item just becomes aware by the user. delta>0 increases prob of purchase/read
delta = 0                       # Factor by which distance decreases for recommended product, 5 default

# Awareness, setting selectes predefined awareness settings
awaremech = 3                   # Awareness mech is wrt: 0=Off, 1=Origin,2=Person,3=Both (note, don't use 2 unless there exists outside good, will make fringe users aware of no products and cause 0 denominator for probability matrix)
theta = 0.35                    # Awareness Scaling, .35 in paper
Lambda = 0.75 					# This is crucial since it controls how much the users focus on mainstream items, 0.75 default value (more focused on mainstream)

# Iterations (for baseline iters1, and with recommenders on iters2)
iters1 = 50                    # Length of period without recommendations (all agents make 1 purchase/iteration)
iters2 = 300                    # Length of period with recommendations (uses sales data left at end of Iters1)
plots = 1                       # 1=produce all plots, 0=plots off

# Added functionalities (compared to Flered's and Hosanagar's), e.g. timer-based awareness, percentage of online products users, moving users (instead of fixed)
added = False
timeValue = 100 				# number of iterations until the awareness fades away, set very high e.g. >iters2 for no effect
percentageOfActiveUsers = 1.0  	# percentage of active users per iteration, set 1 to agree with paper
percentageOfActiveItems = 1.0 	# percentage of active items per iteration, set 1 to agree with paper
moveAsDistancePercentage = 0.01 # the amount of distance covered when a user move towards an item



	
# Generate products and users/customers
Users = np.array([ [-1.5, -1.5] for i in range(A)])

PP= [] 
for i in range(0,30):
	for j in range(0,30):
		p = [i/30*4-2, j/30*4-2 ]
		PP.append(p)
I = len(PP)
Products =np.array(PP)  #np.array([ [random.random()*4-2, random.random()*4-2] for i in range(I)])
varBeta = np.array([0 for i in range(A)]) #np.array([(random.random()*40-20) for i in range(A)])


# Run simulation: configuration 1
added = False
plots = False
Ginis = sim2fcntimeseries(A,I,iters1,iters2,n,delta,Users,Products,engine,metric,k,plots,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda,timeValue,percentageOfActiveUsers, percentageOfActiveItems, added, moveAsDistancePercentage)
#D.append(Ginis)	

	



