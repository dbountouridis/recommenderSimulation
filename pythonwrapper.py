from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
	R = [random.random() for v in range(len(V))]
	E = -np.log(-np.log(R))
	U = V + E
	sel = np.where(w==1)[0]
	mx = np.argmax(U[sel])
	i = sel[mx]
	return i # index of chosen item


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
	    

# Main simulation function
def sim2fcntimeseries(A,I,iters1,iters2,n,delta,Users,Products,engine,metric,k,plots,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda,timer, percentageOfActiveUsers, percentageOfActiveItems):
	
	# Initialize stuctures
	if opdist<=0: 	# if not outer product
		P = np.zeros([A,I]) 	 # Purchases, sales history
		H = np.zeros([A,I]) 	 # Loyalty histories
		InitUsers = Users.copy()


		# Create distance matrix
		D = np.zeros([A,I])
		D = spatial.distance.cdist(Users, Products)					# distance of products from users
		Do = spatial.distance.cdist([[0,0]], Products)[0] 			# distance of products from origin, remains fixed for each engine


		# Make binary awareness matrix 
		W = makeawaremx(awaremech,theta,Products,D,A,I,Lambda)
		

		# Make timer matrix for awareness
		T = W.copy()
		indecesOfInitialAwareness = W==1
		T[T==1] = timeValue
		

		# Make a dictionary structure
		Data = {}
		for eng in engine+["Control"]:
			if eng=="Control": iters = iters1
			else: iters = iters2
			Data.update({eng:{"Product Sales Time Series": np.ones([I, iters]), "Sales History": P.copy(),"All Purchased Products":[],"Users":Users.copy(),"Awareness":W.copy(),"D":D.copy(),"T":T.copy(),"H":H.copy(),"Iterations":iters}})


	# Ginis array
	Ginis = []
	
	
	# Simulation
	for eng in ["Control"]+engine:
		print("Engine ",eng," period...")
		
		if eng is not "Control":
			# continue from the Control period history
			Data[eng]["Sales History"] = Data["Control"]["Sales History"].copy()
			Data[eng]["Product Sales Time Series"][:,0] = Data["Control"]["Product Sales Time Series"][:,-1] 
			Data[eng]["H"] = Data["Control"]["H"]
				
		
		for t in range(Data[eng]["Iterations"]):
			if t>0: Data[eng]["Product Sales Time Series"][:,t] = Data[eng]["Product Sales Time Series"][:,t-1]

			# update user activity: random users that are online
			activeUserIndeces = np.arange(A).tolist()
			random.shuffle(activeUserIndeces)
			activeUserIndeces = activeUserIndeces[:int(len(activeUserIndeces)*percentageOfActiveUsers)] 

			# update products availability: random products that are available
			activeItemIndeces = np.arange(I).tolist()
			random.shuffle(activeItemIndeces)
			activeItemIndeces = np.sort(activeItemIndeces[:int(len(activeItemIndeces)*percentageOfActiveItems)]).tolist()
			nonActiveItemIndeces = [ i  for i in np.arange(I) if i not in activeItemIndeces]
			
			W__ = Data[eng]['Awareness'].copy()
			W__[:,nonActiveItemIndeces] = 0  # make the awareness 0, therefore unavailable

			for a in activeUserIndeces:
				if eng is not "Control":
					Rec = activeItemIndeces[recengine(eng,Data[eng]["Sales History"][:,activeItemIndeces],a,n,opdist)] 	# recommendation
					W__[a,Rec] = 1												# Rec forces permanent awareness in the original implementation
					Data[eng]['T'][a,Rec] = timeValue 							# but we minimize that effect with a timer													
					indexOfChosenItem =  LogitChoiceByHand(Data[eng]["D"][a,:], metric, k, spec, delta, Rec, Data[eng]["H"][a,:], varBeta[a], W__[a,:])
				else:
					indexOfChosenItem = activeItemIndeces[ LogitChoiceByHand(Data[eng]["D"][a,activeItemIndeces], metric, k, 0, delta, 1, Data[eng]["H"][a,activeItemIndeces], varBeta[a], Data[eng]["Awareness"][a,activeItemIndeces]) ]
				Data[eng]["H"][a,:] = varAlpha*Data[eng]["H"][a,:]													# update loyalty smooths
				Data[eng]["H"][a,indexOfChosenItem] +=(1-varAlpha)					# update loyalty smooths
				Data[eng]["Product Sales Time Series"][indexOfChosenItem,t] +=1		# add product sale
				Data[eng]["Sales History"][a,indexOfChosenItem]+=1					# add to sales history, the P matrix in the original code
				Data[eng]["All Purchased Products"].append(indexOfChosenItem)		# add product sale

			# Adjust awareness
			Data[eng]['T'] = Data[eng]['T']-1
			Data[eng]['T'][indecesOfInitialAwareness] = timeValue # make sure the initial awareness does not fade
			Data[eng]['T'][Data[eng]['T']<0] = 0 # make sure there are not negative values
			Data[eng]['Awareness'][T==0] = 0 

			# 	# compute new user location
			# 	B = np.array(Users[a])
			# 	P = np.array(Products[indexOfChosenItem])
			# 	BP = P - B
			# 	x,y = B + 0.01*BP
			# 	Users[a] = [x,y]

			# # update
			# D = spatial.distance.cdist(Users, Products)	# distance of products from users
			# W = makeawaremx(awaremech,theta,Products,D,A,I,Lambda)
		

		# Gini computation
		if eng is not "Control":
			# since the recommendation period started with the purchase data of the control period do the following
			Data[eng]["Sales History"] = Data[eng]["Sales History"] - Data["Control"]["Sales History"]
			
			# Calculate Gini
			G1 = gini(np.sum(Data["Control"]["Sales History"],axis=0))
			G2 = gini(np.sum(Data[eng]["Sales History"],axis=0))
			Ginis.append(G2 - G1)
			print("Gini (control period):",G1, " (Recommender period):", G2)



	# Some plotting for visual inspection
	if plots:
		sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})

		binsI = stats.mstats.mquantiles(Do, [i/20 for i in range(20)]+[1])
		index = [np.argmin(abs(binsI-i))  for i in Do]
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


		f, ax = plt.subplots(1,1+len(engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+engine):
			n, bins = np.histogram(Data[period]["All Purchased Products"], bins=range(I+1))
			circle1 = plt.Circle((0, 0), 1, color='g',fill=True,alpha=0.3,zorder=-1)
			ax[p].add_artist(circle1)
			ax[p].scatter(Users[:,0], Users[:,1], marker='.', c='b',s=40,alpha=0.6)
			#ax[p].scatter(InitUsers[:,0], InitUsers[:,1], marker='.', c='y',s=40,alpha=0.6)
			for i in range(I): 
				if n[i]>=1:
					v = 0.3+ n[i]/np.max(n)*0.7
					c = (1,0,0,v)
					s = 10+n[i]/np.max(n)*30
				else:
					c = (0.2,.2,0.2,0.2)
					s = 10
				ax[p].scatter(Products[i,0], Products[i,1], marker='o', c=c,s=s)		
			ax[p].set_xlabel(period)
			ax[p].set_aspect('equal', adjustable='box')
		plt.tight_layout()
		plt.show()

	return Ginis
		


# Inputs
A = 100                         # Agents, users
I = 100                         # Items, products
engine = ["CF","random","median"]                      
n = 5                           # Top-n similar used in collaborative filter

# Choice model
metric = 2                       # 1=(-1)*Distance, 2= -k*Log(Distance), 3=(1/Distance),  2 from paper
# the higher the k the the consumer prefers closest products
k = 10                           # Constant used in similarity function,  10 from paper
opdist = 0                       # Outside good's distance 0=Off 0.75 in paper

# Variety seeking (Section 7.3)
varAlpha = 0.75                  # Variety: parameter governing exponential smooth, 0.75 in paper
# varBeta > 0 = inertia: consumers don't want to deviate, recommender
# influence limited
# varBeta < 0 = variety seeking:
varBeta = 0                    # Variety: coefficient of smooth history in utility function

# Salience
spec = 3                         # Utility spec for salience: 1=f(del*D), 2=del*f(D), 3=f(D)+del
# if delta is 0, then the item just becomes aware by the user. delta>0 increases prob of purchase/read
delta = 5                       # Factor by which distance decreases for recommended product

# Awareness, setting selectes predefined awareness settings
awaremech = 3                    # Awareness mech is wrt: 0=Off, 1=Origin,2=Person,3=Both (note, don't use 2 unless there exists outside good, will make fringe users aware of no products and cause 0 denominator for probability matrix)
theta = 0.35                        # Awareness Scaling, .35 in paper
Lambda = 0.75 	# This is crucial since it controls how much the users focus on mainstream items, 0.75 default value (more focused on mainstream)

# Iterations (for baseline iters1, and with recommenders on iters2)
iters1 = 100                      # Length of period without recommendations (all agents make 1 purchase/iteration)
iters2 = 100                      # Length of period with recommendations (uses sales data left at end of Iters1)
plots = 1                       # 1=produce all plots, 0=plots off

# Added functionalities
timeValue = 100 	# number of iterations until the awareness fades away, set very high e.g. >iters2 for no effect
percentageOfActiveUsers = 0.9  	# percentage of active users per iteration, set 1 to agree with paper
percentageOfActiveItems = 0.9 	# percentage of active items per iteration, set 1 to agree with paper

D = [] # hold the G2-G1 values
for i in range(1):
	print("Run:",i)
	
	# Generate products and users/customers
	Users = np.array([ [np.random.normal(), np.random.normal()] for i in range(A)])
	Products = np.array([ [np.random.normal(), np.random.normal()] for i in range(I)])
	varBeta = np.array([(random.random()*40-20) for i in range(A)])

	# Run simulation: configuration 1
	Ginis = sim2fcntimeseries(A,I,iters1,iters2,n,delta,Users,Products,engine,metric,k,plots,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda,timeValue,percentageOfActiveUsers, percentageOfActiveItems)
	# D.append(Ginis)	

	# # Run simulation: configuration 2
	# varBeta = np.array([(random.random()*40-20) for i in range(A)])
	# Ginis2 = sim2fcntimeseries(A,I,iters1,iters2,n,delta,Users,Products,engine,metric,k,plots,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda,timeValue,percentageOfActiveUsers, percentageOfActiveItems)
	# D.append(Ginis+Ginis2)	

df = pd.DataFrame(D, columns = [i for i in engine])
print(df)
print(df.describe())
df.plot.box()
plt.show()


