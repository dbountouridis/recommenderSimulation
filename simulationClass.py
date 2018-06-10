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
import networkx as nx


__author__ = 'Dimitrios Bountouridis'


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


class simulation(object):
	def __init__(self):
		# Default settings
		# Inputs
		self.A = 10                         # Agents, users
		self.I = 50                         # Items, products
		self.engine =["CF"]#["CF","min","random"] #["CF","CFnorm","min","random","max","median"]                      
		self.n = 5                           # Top-n similar used in collaborative filter

		# Choice model
		self.metric = 2                      # 1=(-1)*Distance, 2= -k*Log(Distance), 3=(1/Distance),  2 from paper
		# the higher the k the the consumer prefers closest products
		self.k = 10                          # Constant used in similarity function,  10 from paper

		# Variety seeking 
		self.varAlpha = 0.75                 # Variety: parameter governing exponential smooth, 0.75 in paper
		self.varBeta = 0                     # Variety: coefficient of smooth history in utility function

		# Salience
		self.spec = 3                        # Utility spec for salience: 1=f(del*D), 2=del*f(D), 3=f(D)+del
		self.delta = 5                       # Factor by which distance decreases for recommended product, 5 default

		# Awareness, setting selectes predefined awareness settings
		self.awaremech = 3                   # Awareness mech is wrt: 0=Off, 1=Origin,2=Person,3=Both (note, don't use 2 unless there exists outside good, will make fringe users aware of no products and cause 0 denominator for probability matrix)
		self.theta = 0.35                    # Awareness Scaling, .35 in paper
		self.Lambda = 0.75 					# This is crucial since it controls how much the users focus on mainstream items, 0.75 default value (more focused on mainstream)

		# Iterations (for baseline iters1, and with recommenders on iters2)
		self.iters1 = 50                    # Length of period without recommendations (all agents make 1 purchase/iteration)
		self.iters2 = 50                    # Length of period with recommendations (uses sales data left at end of Iters1)

		# Added functionalities (compared to Flered's and Hosanagar's), e.g. timer-based awareness, percentage of online products users, moving users (instead of fixed)
		self.added = False
		self.timeValue = 100 				# number of iterations until the awareness fades away, set very high e.g. >iters2 for no effect
		self.percentageOfActiveUsers = 0.5  	# percentage of active users per iteration, set 1 to agree with paper
		self.percentageOfActiveItems = 1.0 	# percentage of active items per iteration, set 1 to agree with paper
		self.moveAsDistancePercentage = 0.05 # the amount of distance covered when a user move towards an item

	# Create an instance of simulation based on the parameters
	def createSimulationInstance(self):
		
		# Generate products and users/customers
		self.Users = np.array([ [np.random.normal()/1.2, np.random.normal()/1.2] for i in range(self.A)])
		self.Products = np.array([ [np.random.normal(), np.random.normal()] for i in range(self.I)])
		self.varBeta = np.array([0 for i in range(self.A)]) #np.array([(random.random()*40-20) for i in range(A)])

		P = np.zeros([self.A,self.I]) 	# Purchases, sales history
		H = P.copy() 	 				# Loyalty histories
	
		# Create distance matrices
		D = spatial.distance.cdist(self.Users, self.Products)			# distance of products from users
		self.Do = spatial.distance.cdist([[0,0]], self.Products)[0] 	# distance of products from origin, remains fixed for each engine

		# Create binary awareness matrix 
		W = self.makeawaremx(D)
		
		# Create timer matrix for awareness
		T = W.copy()
		indecesOfInitialAwareness = W==1
		T[indecesOfInitialAwareness] = self.timeValue

		# Create a dictionary structure
		self.Data = {}
		for eng in self.engine+["Control"]:
			if eng=="Control": iters = self.iters1
			else: iters = self.iters2
			self.Data.update({eng:{"Product Sales Time Series" : np.ones([self.I, iters]), "Sales History" : P.copy(),"All Purchased Products" : [],"Users" : self.Users.copy(),"InitialUsers" : self.Users.copy(),"Awareness" : W.copy(),"D" : D.copy(),"T" : T.copy(),"H" : H.copy(),"Iterations" : iters,"X" : np.zeros([self.A,iters]),"Y" : np.zeros([self.A,iters])}})

	# Make awareness matrix
	def makeawaremx(self,Dij):
		# awaremech==3   # not used currently
		W = np.zeros([self.A,self.I])
		for a in range(self.A):
			for i in range(self.I):
				W[a,i] = self.Lambda*np.exp(-(np.power(self.Do[i],2))/(self.theta/1)) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/(self.theta/3))
				W[a,i] = random.random()<W[a,i] # probabilistic
		return W

	# Probabilistic choice model
	def ChoiceModel(self, eng, user, Rec, w, control = False):

		Distances = self.Data[eng]["D"][user,:]
		smoothhist = self.Data[eng]["H"][user,:]
		varBeta =  self.varBeta[user]

		if control : spec = 0 # if control period
		else: spec = self.spec

		if spec==1: Distances[Rec] = Distances[Rec]*self.delta

		# Convert distance to similarity based on metric
		if self.metric == 1: Similarity = - Distances
		if self.metric == 2: Similarity = - self.k*np.log(Distances)
		if self.metric == 3: Similarity = np.power(Distances,-self.k)

		# Calc deterministic utility (based on utility spec desired)
		#spec: 0 = f(d), 1 = f(Delta*d), 2 = delta*f(d), 3 = f(d) + Delta
		V = 1*Similarity + varBeta*smoothhist

		# If spec==0, f(d)      +...  don't do anything: rec's off and no salience
		# If spec==1, f(Delta*d)+...  don't do anything: already multiplied above
		if spec==2:  
			V[Rec] = self.delta*1*Similarity[Rec] + varBeta*smoothhist[Rec]
		if spec==3:
			V[Rec] = 1*Similarity[Rec] + self.delta + varBeta*smoothhist[Rec]
		
		# Introduce the stochastic component
		R = [random.random() for v in range(len(V))]
		E = -np.log(-np.log(R))
		U = V + E
		sel = np.where(w==1)[0]
		mx = np.argmax(U[sel])
		i = sel[mx]
		return i # index of chosen item

	# Compute new position of a user given they purchased an item
	def computeNewPositionOfUserToItem(self, eng, user, indexOfChosenItem, iteration):
		# compute new user location. But the probability that the user will move towards
		# the item is proportional to their distance
		dist = self.Data[eng]["D"][user,indexOfChosenItem]
		p = np.exp(-(np.power(dist,2))/(.35/3)) # based on the awareness forumla
		B = np.array(self.Data[eng]["Users"][user])
		P = np.array(self.Products[indexOfChosenItem])
		BP = P - B
		x,y = B + self.moveAsDistancePercentage*(random.random()<p)*BP 	# probabilistic
		self.Data[eng]["Users"][user] = [x,y]
		self.Data[eng]["X"][user,iteration] = x
		self.Data[eng]["Y"][user,iteration] = y

	# Extra functionalities that extend Fleder and Hosanagar, applied at the end of each iteration
	def addedFunctionalitiesAfterIteration(self, eng):
		# # Adjust awareness based on timer
		# Data[eng]['T'] = Data[eng]['T']-1
		# #Data[eng]['T'][indecesOfInitialAwareness] = timeValue # make sure the initial awareness does not fade
		# Data[eng]['T'][Data[eng]['T']<0] = 0 # make sure there are not negative values
		# Data[eng]['Awareness'][T==0] = 0 
		# Data[eng]['Awareness'][T!=0] = 1 
		#print("b:",np.sum(Data[eng]['Awareness'].flatten()))

		# update distances and awereness based on new positions
		self.Data[eng]["D"] = spatial.distance.cdist(self.Data[eng]["Users"], self.Products)	# distance of products from users
		newAwareness = self.makeawaremx(self.Data[eng]["D"])
		indeces = newAwareness==1
		self.Data[eng]['T'][indeces] = self.timeValue

		self.Data[eng]['Awareness']=newAwareness
		indeces = self.Data[eng]['Awareness']>1
		self.Data[eng]["Awareness"][indeces]=1
		#print("a:",np.sum(Data[eng]['Awareness'].flatten()))

	# If extra funcionalities are active, return a random subset of users and items per iteration
	def generateRandomSubsetOfAvailableUsersItems(self):
		
		# update user, product availability: random users that are online
		activeUserIndeces = np.arange(self.A).tolist()
		activeItemIndeces = np.arange(self.I).tolist()
		nonActiveItemIndeces = [ i  for i in np.arange(self.I) if i not in activeItemIndeces]

		if not self.added: return (activeUserIndeces, activeItemIndeces, nonActiveItemIndeces) 

		random.shuffle(activeUserIndeces)
		random.shuffle(activeItemIndeces)
		activeUserIndeces = activeUserIndeces[:int(len(activeUserIndeces)*self.percentageOfActiveUsers)] 
		activeItemIndeces = np.sort(activeItemIndeces[:int(len(activeItemIndeces)*self.percentageOfActiveItems)]).tolist()
		nonActiveItemIndeces = [ i  for i in np.arange(self.I) if i not in activeItemIndeces]
		nonActiveUserIndeces = [ i  for i in np.arange(self.A) if i not in activeUserIndeces]

		return (activeUserIndeces, nonActiveUserIndeces, activeItemIndeces, nonActiveItemIndeces) 

	# Run the simulation
	def runSimulation(self):
		
		for eng in ["Control"]+self.engine: 	# start from the control period
			print("* Engine ",eng," period...")
			
			if eng is not "Control":
				# continue from the Control period history
				self.Data[eng]["Sales History"] = self.Data["Control"]["Sales History"].copy()
				self.Data[eng]["Product Sales Time Series"][:,0] = self.Data["Control"]["Product Sales Time Series"][:,-1] 
				self.Data[eng]["H"] = self.Data["Control"]["H"].copy()
				self.Data[eng]["InitialUsers"] = self.Data["Control"]["Users"].copy() 	# this won't be updated
				self.Data[eng]["Users"] = self.Data["Control"]["Users"].copy()			# this will be updated
			
			# for each iteration
			for t in range(self.Data[eng]["Iterations"]):
				if t>0: 
					self.Data[eng]["Product Sales Time Series"][:,t] = self.Data[eng]["Product Sales Time Series"][:,t-1]
					self.Data[eng]["X"][:,t] = self.Data[eng]["X"][:,t-1]
					self.Data[eng]["Y"][:,t] = self.Data[eng]["Y"][:,t-1]
					print([item for item in zip(self.Data[eng]["X"],self.Data[eng]["Y"])])

				# random subset of available users and items, applied when added = True
				(activeUserIndeces, nonActiveUserIndeces, activeItemIndeces, nonActiveItemIndeces) = self.generateRandomSubsetOfAvailableUsersItems()
				
				# adjust awareness for only available items
				W__ = self.Data[eng]['Awareness'].copy()
				W__[:,nonActiveItemIndeces] = 0  
				indecesOfInitialAwareness = W__==1

				# compute item choice for each active user
				for user in activeUserIndeces:
					if eng is not "Control":
						Rec = activeItemIndeces[self.recengine(eng, user, activeItemIndeces)] 	# recommendation
						W__[user, Rec] = 1														# Rec forces permanent awareness in the original implementation
						self.Data[eng]['T'][user, Rec] = self.timeValue 						# but we minimize that effect with a timer													
						indexOfChosenItem =  self.ChoiceModel(eng, user, Rec, W__[user,:])
					else:
						indexOfChosenItem =  self.ChoiceModel(eng, user, -1, W__[user,:], control = True) 
					
					self.Data[eng]["H"][user,:] = self.varAlpha*self.Data[eng]["H"][user,:]													# update loyalty smooths
					self.Data[eng]["H"][user, indexOfChosenItem] += (1-self.varAlpha)			# update loyalty smooths
					self.Data[eng]["Product Sales Time Series"][indexOfChosenItem,t] += 1	# add product sale
					self.Data[eng]["Sales History"][user, indexOfChosenItem] += 1				# add to sales history, the P matrix in the original code
					self.Data[eng]["All Purchased Products"].append(indexOfChosenItem)		# add product sale

					# if added is True, compute new user position
					if self.added : self.computeNewPositionOfUserToItem(eng, user, indexOfChosenItem, t)
				
				# if added is True, do the following after each iteration	
				if self.added:  self.addedFunctionalitiesAfterIteration(eng)
					
	# Recommendation algorithms (engines)
	def recengine(self, engine, a, activeItemIndeces):

		P = self.Data[engine]["Sales History"][:,activeItemIndeces]

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
				topN = np.argsort(s)[::-1][:self.n] # top N users
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
				topN = np.argsort(s)[::-1][:self.n] # top N users
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

	# Diversity measure: gini coefficients
	# based on: Kartik Hosanagar, Daniel Fleder (2008)
	def computeGinis(self):
		GiniPerRec = {}
		
		for eng in self.engine:
			# since the recommendation period started with the purchase data of the control period do the following
			self.Data[eng]["Sales History"] = self.Data[eng]["Sales History"] - self.Data["Control"]["Sales History"]
			G1 = gini(np.sum(self.Data["Control"]["Sales History"],axis=0))
			G2 = gini(np.sum(self.Data[eng]["Sales History"],axis=0))
			GiniPerRec.update({eng:G2 - G1})

		return GiniPerRec
			
	# Plotting	    
	def plot2D(self):
		sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})

		# plot users, products on 2d plane
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			n, bins = np.histogram(self.Data[period]["All Purchased Products"], bins=range(self.I+1))
			circle1 = plt.Circle((0, 0), 1, color='g',fill=True,alpha=0.3,zorder=-1)
			ax[p].add_artist(circle1)
			#ax[p].scatter(self.Data[period]["Initialself.Users"][:,0], self.Data[period]["Initialself.Users"][:,1], marker='.', c='b',s=40,alpha=0.3)
			for i in range(len(self.Users[:,1])):
				ax[p].scatter(self.Data[period]["Users"][i,0], self.Data[period]["Users"][i,1], marker='.', c='b',s=10, alpha = 0.6 )
			for i in range(len(self.Users[:,1])):
				for j in range(len(self.Data[period]["X"][0,:])-1):
					ax[p].plot([self.Data[period]["X"][i,j], self.Data[period]["X"][i,j+1]], [self.Data[period]["Y"][i,j], self.Data[period]["Y"][i,j+1]], 'b-', lw=0.5, alpha =0.6)
			#ax[p].scatter(Initself.Users[:,0], Initself.Users[:,1], marker='.', c='y',s=40,alpha=0.6)
			for i in range(self.I): 
				if n[i]>=1:
					v = 0.6# 0.4+ n[i]/np.max(n)*0.4
					c = (1,0,0.0,v)
					s = 2+n[i]/np.max(n)*40
					marker='o'
				else:
					c = (0,0,0,0.8)
					s = 10
					marker='x'
				ax[p].scatter(self.Products[i,0], self.Products[i,1], marker=marker, c=c,s=s)		
			ax[p].set_xlabel(period)
			ax[p].set_aspect('equal', adjustable='box')
		plt.tight_layout()
		plt.savefig("plots/users-products.pdf")
		plt.show()

		# plot only the users
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			n, bins = np.histogram(self.Data[period]["All Purchased Products"], bins=range(self.I+1))
			circle1 = plt.Circle((0, 0), 1, color='g',fill=True,alpha=0.3,zorder=-1)
			ax[p].add_artist(circle1)
			#ax[p].scatter(self.Data[period]["Initialself.Users"][:,0], self.Data[period]["Initialself.Users"][:,1], marker='.', c='b',s=40,alpha=0.3)
			for i in range(len(self.Users[:,1])):
				ax[p].scatter(self.Data[period]["Users"][i,0], self.Data[period]["Users"][i,1], marker='.', c='b',s=40,alpha=0.2+0.8*((self.varBeta[i]+20)/40) )
			ax[p].set_xlabel(period)
			ax[p].set_aspect('equal', adjustable='box')
		plt.tight_layout()
		plt.savefig("plots/users.pdf")
		plt.show()

		# histogram of user distances (to investigate their clusterdness)
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			x = spatial.distance.cdist(self.Data[period]["Users"], self.Data[period]["Users"])
			sns.distplot(x.flatten(),ax=ax[p])
			ax[p].set_xlabel(period)
		plt.savefig("plots/users-dist-distribution.pdf")
		plt.show()

	# Network based measures of fragmentation
	# based on: Kartik Hosanagar, Daniel Fleder, Dokyun Lee, Andreas Buja (2014)
	def networkAnalysis(self):
		
		MedianDegrees = {}

		if not self.added:
			print("User adaptability not active, returning empty dictionary.")
			return MedianDegrees

		for eng in ["Control"]+self.engine:
			# convert distance matrix to similarity
			k = 10
			matrix = - k*np.log(spatial.distance.cdist(self.Data[eng]["Users"], self.Data[eng]["Users"]))
			
			# network 
			G = nx.Graph()

			# create nodes
			nodes = range(0,matrix.shape[0])
			for node in nodes: G.add_node(node)

			# add edges
			for node1 in range(0,matrix.shape[0]):
				for node2 in range(node1+1,matrix.shape[0]):
					G.add_edge(node1, node2,weight=matrix[node1,node2])

			# Median degree
			medianDegree = np.mean([item[1] for item in G.degree(weight='weight').items()])
			MedianDegrees.update({eng: medianDegree})

			G.clear()
				
		return MedianDegrees

# Run simulations
D = [] # hold the G2-G1 values
for i in range(1):
	print("Run:",i)
	
	print("Initialize simulation class...")
	sim = simulation()
	sim.added = True
	print("Create simulation instance...")
	sim.createSimulationInstance()
	print("Run simulation...")
	sim.runSimulation()
	print("=== Analysis ===")
	print(" Gini coefficients per rec engine:")
	g = sim.computeGinis()
	print(g)
	print(" Network analytics:")
	m = sim.networkAnalysis()
	print(m)
	print("Plotting...")
	sim.plot2D()




# df = pd.DataFrame(D, columns = [i for i in engine]+[str(i)+"-v2" for i in engine])
# print(df)
# print(df.describe())
# df.plot.box()
# plt.show()


