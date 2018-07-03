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
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture


__author__ = 'Dimitrios  Bountouridis'


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
		self.A = 30                        # Agents, users
		self.I = 3500                       # Items, products
		self.engine = []#["CF","CFnorm","min","random"]                      
		self.n = 5                          # Top-n similar used in collaborative filter
		
		# Choice model
		# the higher the k the the consumer prefers closest products
		self.k = 10                          # Constant used in similarity function,  10 from paper

		# Salience
		self.delta = 5                       # Factor by which distance decreases for recommended product, 5 default

		# Awareness, setting selectes predefined awareness settings
		self.theta = 0.1                    # Awareness Scaling, .35 in paper
		self.thetaDot = 0.065
		self.Lambda = 0.5 					# This is crucial since it controls how much the users focus on mainstream items, 0.75 default value (more focused on mainstream)
		self.rankingDecreasePerIteration = 0.1

		# Iterations (for baseline iters1, and with recommenders on iters2)
		self.iters1 = [i for i in range(19)]        # Length of period without recommendations (all agents make 1 purchase/iteration)
		self.iters2 = [i for i in range(10,20)]     # Length of period with recommendations (uses sales data left at end of Iters1)        
		self.percentageOfActiveUsers = 1.0      # percentage of active users per iteration, set 1 to agree with paper 
		self.percentageOfActiveItems = 0.032     # percentage of active items per iteration, set 1 to agree with paper         
		self.moveAsDistancePercentage = 0.1    # the amount of distance covered when a user move towards an item 
		self.userVarietySeeking = []
		self.categories = ["business", "entertainment", "politics", "sport", "tech"]          
		self.categoriesSalience = [1,0.7,0.8,0.6,0.9] # arbitrary assigned
		self.Pickle = []

	# Create an instance of simulation based on the parameters
	def createSimulationInstance(self, seed = None):
		random.seed(seed)

		''' 
			Item related paremeterizations:
			Items are grouped into classes corresponding to their news topic. Currently they are generated
			as 2d guassian distributions roughly centered around (0, 0). Later their 2d distribution should
			be based on PCA/t-sne feature reduction on real life data. New items appear at each iteration 
			but instead of generating new items we generate all of them at the beginning, and adjust their 
			availability at each iteration. Items also have a limited lifespan. Items have energy also.
		'''
		# generate items products
		(X,labels,topicClasses) = pickle.load(open('BBC data/t-SNE-projection.pkl','rb'))
		print(set(labels))
		gmm = GaussianMixture(n_components=5).fit(X)
		samples_,self.ItemsClass = gmm.sample(self.I)
		self.Items = samples_/55  # scale down to -1, 1 range
		self.ItemFeatures = gmm.predict_proba(samples_)

		# generate a random order of item availability
		self.itemOrderOfAppearance = np.arange(self.I).tolist()
		random.shuffle(self.itemOrderOfAppearance)

		# create timer matrix (lifespan) for item availability
		L = np.array([1 for i in range(self.I)])

		# distance of products from origin, remains fixed for each engine
		# self.Do = spatial.distance.cdist([[0,0]], self.Items)[0] 

		# item ranking
		self.initialR = np.zeros(self.I)
		for i in range(5):
			indeces = np.where(self.ItemsClass==i)[0]
			lower, upper = 0, 1
			mu, sigma = self.categoriesSalience[i], 0.15
			X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
			self.initialR[indeces] = X.rvs(len(indeces))
			# fig, ax = plt.subplots(2, sharex=True)
			# ax[0].hist(self.initialR[indeces], normed=True)
			# plt.show()
			
			 	
		''' 
			User related paremeterizations:
			Currently users are generated as a 2d gaussian around the center (0, 0). A uniform distribution
			should be later used.
		'''
		# Generate users/customers
		self.Users = np.random.uniform(-1,1,(self.A,2))

		# size of session per user (e.g. amount of articles read per day)
		self.UserSessionSize = [1 for i in range(self.A)]
		print(self.UserSessionSize)

		# Randomly assign how willing each user is to change preferences (used in choice model). 
		# Normal distribution centered around 0.5
		lower, upper = 0, 1
		mu, sigma = 0.05, 0.05
		X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		self.userVarietySeeking = X.rvs(self.A)
		# print(self.userVarietySeeking)
		# fig, ax = plt.subplots(2, sharex=True)
		# ax[0].hist(self.userVarietySeeking, normed=True)
		# plt.show()


		P = np.zeros([self.A,self.I]) 	# Purchases, sales history
		H = P.copy() 	 				# Loyalty histories
	
		# Create distance matrices
		D = spatial.distance.cdist(self.Users, self.Items)			# distance of products from users
		
		# Store everything in a dictionary structure
		self.Data = {}
		for eng in self.engine+["Control"]:
			if eng=="Control": iters = self.iters1
			else: iters = self.iters2
			self.Data.update({eng:{"ItemRanking":self.initialR.copy(), "Sales History" : P.copy(),"All Purchased Items" : [],"Users" : self.Users.copy(),"InitialUsers" : self.Users.copy(),  "D" : D.copy(),"ItemLifespan" : L.copy(),"H" : H.copy(),"Iterations" : iters,"X" : np.zeros([self.A,len(iters)]),"Y" : np.zeros([self.A,len(iters)])}})
			self.Data[eng]["X"][:,0] = self.Data[eng]["Users"][:,0]
			self.Data[eng]["Y"][:,0] = self.Data[eng]["Users"][:,1]

	# Make awareness matrix
	def makeawaremx(self,eng):
		random.seed(1)

		Dij = self.Data[eng]["D"]
		W = np.zeros([self.A,self.I])
		W2 = W.copy() # for analysis purposes
		W3 = W.copy()
		for a in range(self.A):
			for i in range(self.I):
				W[a,i] = self.Lambda*np.exp(-(1/self.thetaDot)*(np.power(1-self.Data[eng]["ItemRanking"][i],2))) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				W2[a,i] = self.Lambda*np.exp(-(1/self.thetaDot)*  (np.power(1-self.Data[eng]["ItemRanking"][i],2))) 
				W3[a,i] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				r = random.random()
				W[a,i] = r<W[a,i] # probabilistic
				# W2[a,i] = r<W2[a,i] # probabilistic
				# W3[a,i] = r<W3[a,i] # probabilistic
		return W,W2,W3

	# Probabilistic choice model
	def ChoiceModel(self, eng, user, Rec, w, control = False):

		# Distances = self.Data[eng]["D"][user,:]
		# Similarity = np.power(Distances,-self.k)
		#Similarity = -self.k*np.log(Distances).  # this one!!!
		# V = 1*Similarity #+ varBeta*smoothhist

		# if not control: V[Rec] = 1*Similarity[Rec] + self.delta 
		
		# # Introduce the stochastic component
		# R = [random.random() for v in range(len(V))]
		# E = -np.log(-np.log(R))
		# U = V + E
		# sel = np.where(w==1)[0]
		# mx = np.argmax(U[sel])
		# i = sel[mx]
		if random.random()<0.3 and w[Rec]==1 and not control: return Rec

		# random choice
		sel = np.where(w==1)[0].tolist()
		random.shuffle(sel)
		return sel[0] # index of chosen item

	# Compute new position of a user given they purchased an item
	def computeNewPositionOfUserToItem(self, eng, user, indexOfChosenItem, iteration):
		# compute new user location. But the probability that the user will move towards
		# the item is proportional to their distance
		dist = self.Data[eng]["D"][user,indexOfChosenItem]
		p = np.exp(-(np.power(dist,2))/(self.userVarietySeeking[user])) # based on the awareness formula
		B = np.array(self.Data[eng]["Users"][user])
		P = np.array(self.Items[indexOfChosenItem])
		BP = P - B
		x,y = B + self.moveAsDistancePercentage*(random.random()<p)*BP 	# probabilistic
		self.Data[eng]["Users"][user] = [x,y]
		self.Data[eng]["X"][user,iteration] = x
		self.Data[eng]["Y"][user,iteration] = y


	'''
		After each iteration the new user to item position is recomputed. Each item's lifespan is also
		decreased (applies only to those items that were available in the current iteration).
	'''
	def rankingFunction(self, currentRanking, life):
		x = life
		y = (-0.1*x+1)*currentRanking
		return max([y, 0])

	def addedFunctionalitiesAfterIteration(self, eng, activeItemIndeces):

		# update user-item distances based on new user positions
		self.Data[eng]["D"] = spatial.distance.cdist(self.Data[eng]["Users"], self.Items)	

		# update lifespan of available items
		self.Data[eng]["ItemLifespan"][activeItemIndeces] = self.Data[eng]["ItemLifespan"][activeItemIndeces]+1
		#self.Data[eng]["ItemLifespan"][self.Data[eng]["ItemLifespan"]<0] = 0 # no negatives	
		# update ranking based on lifespan, naive
		for a in activeItemIndeces:
			self.Data[eng]['ItemRanking'][a]= self.rankingFunction(self.initialR[a],self.Data[eng]['ItemLifespan'][a])


	'''
		At the beginning of each iteration only a number of users can be available (currently all of them). 
		At each iteration new items are becoming available and those with expired lifespan become unavailable.
	'''
	def subsetOfAvailableUsersItems(self,iteration, eng):
		
		# user availability
		activeUserIndeces = np.arange(self.A).tolist()
		random.shuffle(activeUserIndeces)
		activeUserIndeces = activeUserIndeces[:int(len(activeUserIndeces)*self.percentageOfActiveUsers)] 
		nonActiveUserIndeces = [ i  for i in np.arange(self.A) if i not in activeUserIndeces]

		# items are gradually (at each iteration) becoming available, but have limited lifspan
		activeItemIndeces =[j for j in self.itemOrderOfAppearance[:(iteration+1)*int(self.I*self.percentageOfActiveItems)] if self.Data[eng]["ItemRanking"][j]>0]
		nonActiveItemIndeces = [ i  for i in np.arange(self.I) if i not in activeItemIndeces]
		
		return (activeUserIndeces, nonActiveUserIndeces, activeItemIndeces, nonActiveItemIndeces) 

	# Run the simulation
	def runSimulation(self):
		
		for eng in ["Control"]+self.engine: 	# start from the control period
			print("========= Engine ",eng," ========")
			
			if eng is not "Control":
				# continue from the Control period history
				self.Data[eng]["Sales History"] = self.Data["Control"]["Sales History"].copy()
				self.Data[eng]["H"] = self.Data["Control"]["H"].copy()
				self.Data[eng]["InitialUsers"] = self.Data["Control"]["Users"].copy() 	# this won't be updated
				self.Data[eng]["Users"] = self.Data["Control"]["Users"].copy()			# this will be updated
				self.Data[eng]["ItemLifespan"] = self.Data["Control"]["ItemLifespan"].copy() # this will be updated
				self.Data[eng]["ItemRanking"] = self.Data["Control"]["ItemRanking"].copy()# this will be updated
			
			# for each iteration
			for epoch_index, epoch in enumerate(self.Data[eng]["Iterations"]):
				print(" Epoch",epoch," epoch index:", epoch_index)
				
				if epoch_index>0: 
					self.Data[eng]["Users"][user]  = self.Data[eng]["Users"][user].copy()
					self.Data[eng]["X"][:,epoch_index] = self.Data[eng]["X"][:,epoch_index-1]
					self.Data[eng]["Y"][:,epoch_index] = self.Data[eng]["Y"][:,epoch_index-1]
					
				# random subset of available users . Subset of available items to all users
				(activeUserIndeces, nonActiveUserIndeces, activeItemIndeces, nonActiveItemIndeces) = self.subsetOfAvailableUsersItems(epoch, eng)
				
				# # compute awareness per user and adjust for availability 
				Awareness, AwarenessOnlyPopular,AwarenessProximity = self.makeawaremx(eng)
				Awareness[:,nonActiveItemIndeces] = 0 

				# # do not make available items that a user has purchased before
				# Before = Awareness.copy()
				Awareness = Awareness - self.Data[eng]["Sales History"]>0

				for i in range(1,10):
					indeces = np.where(self.Data[eng]["ItemLifespan"]==i)[0]
					A = Awareness[:,indeces]
					print("    Mean #aware of age",i," :",np.mean(np.sum(A,axis=1))/np.mean(np.sum(Awareness,axis=1)) )
				 
				
				# print("    Mean #aware of:",np.mean(np.sum(Awareness,axis=1)),"Mean #aware of before:",np.mean(np.sum(Before,axis=1)))
				# print("    Mean #aware of popularity:",np.mean(np.sum(AwarenessOnlyPopular,axis=1)),"Mean #aware of proximity:",np.mean(np.sum(AwarenessProximity,axis=1)))
				# print("    Mean lifespan of available items:",np.mean(self.Data[eng]["ItemLifespan"][activeItemIndeces]))
				# print('    Available and non available:',len(activeItemIndeces),len(nonActiveItemIndeces))
				

				# for each active user
				for user in activeUserIndeces:
					# if epoch>1: self.simplePlot(forUser = user, awareness = Awareness[user, :], active = activeItemIndeces)

					# the recommendation stays the same for every session per user	
					if eng is not "Control":
						# recommend one of the available items
						Rec = activeItemIndeces[self.recengine(eng, user, activeItemIndeces)] 	
						# temporary adjust awareness for that item-user pair
						Awareness[user, Rec] = 1				

						# if the user has already purchased the item then decrease awareness of the recommendation
						if 	self.Data[eng]["Sales History"][user,Rec]>0: Awareness[user, Rec] = 0		
						
					# for each article to be read in the user's session
					for readingItem in range(self.UserSessionSize[user]): 
						
						# pick article to read
						if eng == "Control": Rec=-1
						indexOfChosenItem =  self.ChoiceModel(eng, user, Rec, Awareness[user,:], control = eng=="Control")

						# make sure it's not read again in that session
						Awareness[user, indexOfChosenItem] = 0

						# add item purchase to histories
						self.Data[eng]["Sales History"][user, indexOfChosenItem] += 1		
						self.Data[eng]["All Purchased Items"].append(indexOfChosenItem)		

						# compute new user position (we don't store the position in the session, only after it is over)
						self.computeNewPositionOfUserToItem(eng, user, indexOfChosenItem, epoch_index)

						# store some data for stats
						# typeOf = "None"
						# if indexOfChosenItem in np.where(AwarenessProximity[user,:]==1)[0]: typeOf = "inProximity"
						# if indexOfChosenItem in np.where(AwarenessOnlyPopular[user,:]==1)[0]: typeOf = "inPopular"
						if epoch>9:
							self.Pickle.append([epoch_index, user, eng ,indexOfChosenItem,self.Data[eng]["ItemLifespan"][indexOfChosenItem], self.Data[eng]["ItemRanking"][indexOfChosenItem],self.ItemsClass[indexOfChosenItem],indexOfChosenItem == Rec ])
						
				# after each iteration	
				self.addedFunctionalitiesAfterIteration(eng, activeItemIndeces)

		# store
		df = pd.DataFrame(self.Pickle,columns=["iteration","userid","engine","itemid","lifespan","inverseSalience","class","wasRecommended"])
		df.to_pickle("temp/history.pkl")
		pickle.dump(self.Data, open("temp/Data.pkl", "wb"))
					
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


	'''
		Diversity measure: gini coefficients
		based on: Kartik Hosanagar, Daniel Fleder (2008)
	'''
	def computeGinis(self):
		GiniPerRec = {}
		
		for eng in self.engine:
			# since the recommendation period started with the purchase data of the control period do the following
			self.Data[eng]["Sales History"] = self.Data[eng]["Sales History"] - self.Data["Control"]["Sales History"]
			G1 = gini(np.sum(self.Data["Control"]["Sales History"],axis=0))
			G2 = gini(np.sum(self.Data[eng]["Sales History"],axis=0))
			GiniPerRec.update({eng:G2 - G1})

		return GiniPerRec
	
	# plot initial users, products on 2d plane 
	def simplePlot(self, forUser = False, awareness = False, active = False):
		sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 1.0})
		sns.set(style="whitegrid")
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		flatui = sns.color_palette("husl", 5)
		f, ax = plt.subplots(1,1, figsize=(6,6))
		if not forUser:
			for i in range(self.A):
				ax.scatter(self.Users[i,0], self.Users[i,1], marker='+', c='b',s=40,alpha=self.userVarietySeeking[i],edgecolors="k",linewidths=0.5)
			for i in range(self.I):
				color = flatui[self.ItemsClass[i]]
				ax.scatter(self.Items[i,0], self.Items[i,1], marker='o', c=color,s=20*self.initialR[i], alpha =0.8)	
		else:
			ax.scatter(self.Users[forUser,0], self.Users[forUser,1], marker='+', c='b',s=40,alpha=self.userVarietySeeking[forUser],edgecolors="k",linewidths=0.5)
			#print(np.where(np.array(awareness)==1)[0])
			for i in np.where(np.array(awareness)==1)[0]:
				color = flatui[self.ItemsClass[i]]
				ax.scatter(self.Items[i,0], self.Items[i,1], marker='o', c=color,s=20*self.initialR[i], alpha =1)	
			for i in np.where(np.array(awareness)==0)[0]:
				if i in active:
					ax.scatter(self.Items[i,0], self.Items[i,1], marker='o', c='k',s=10*self.initialR[i], alpha =0.1)	

		ax.set_aspect('equal', adjustable='box')
		plt.tight_layout()
		plt.savefig("plots/initial-users-products.pdf")
		plt.show()

	# Plotting	    
	def plot2D(self, drift = False):
		self.Data = pickle.load(open("temp/Data.pkl",'rb'))

		sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		flatui = sns.color_palette("husl", 8)

		# plot users, products on 2d plane
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			n, bins = np.histogram(self.Data[period]["All Purchased Items"], bins=range(self.I+1))

			# final user position as a circle
			for i in range(len(self.Users[:,1])):
				ax[p].scatter(self.Data[period]["Users"][i,0], self.Data[period]["Users"][i,1], marker='.', c='b',s=10, alpha = 0.6 )
			
			# user drift
			if drift:
				for i in range(len(self.Users[:,1])):
					for j in range(len(self.Data[period]["X"][0,:])-1):
						ax[p].plot([self.Data[period]["X"][i,j], self.Data[period]["X"][i,j+1]], [self.Data[period]["Y"][i,j], self.Data[period]["Y"][i,j+1]], 'b-', lw=0.5, alpha =0.6)

			# products
			for i in range(self.I): 
				color = flatui[self.ItemsClass[i]]
				if n[i]>=1:
					v = 0.4+ n[i]/np.max(n)*0.4
					c = (1,0,0.0,v)
					s = 2+n[i]/np.max(n)*40
					marker = 'o'
				else:
					color = (0,0,0,0.1)
					v = 0.1
					s = 2
					marker = 'x'
				ax[p].scatter(self.Items[i,0], self.Items[i,1], marker=marker, c=color,s=s,alpha=v)		
			ax[p].set_xlabel(period)
			ax[p].set_aspect('equal', adjustable='box')
		plt.tight_layout()
		plt.savefig("plots/users-products.pdf")
		plt.show()

		# plot only the users
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			n, bins = np.histogram(self.Data[period]["All Purchased Items"], bins=range(self.I+1))
			circle1 = plt.Circle((0, 0), 1, color='g',fill=True,alpha=0.3,zorder=-1)
			ax[p].add_artist(circle1)
			#ax[p].scatter(self.Data[period]["initialRelf.Users"][:,0], self.Data[period]["initialRelf.Users"][:,1], marker='.', c='b',s=40,alpha=0.3)
			for i in range(len(self.Users[:,1])):
				ax[p].scatter(self.Data[period]["Users"][i,0], self.Data[period]["Users"][i,1], marker='.', c='b',s=40,alpha=0.2+0.8*((self.varBeta[i]+20)/40) )
			ax[p].set_xlabel(period)
			ax[p].set_aspect('equal', adjustable='box')
		plt.tight_layout()
		plt.savefig("plots/users.pdf")
		plt.show()

		# histogram of euclidean user distances (to investigate their clusterdness)
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			x = spatial.distance.cdist(self.Data[period]["Users"], self.Data[period]["Users"], metric = "euclidean")
			iu1 = np.triu_indices(self.A,1)
			x = x[iu1]
			sns.distplot(x.flatten(),ax=ax[p])
			ax[p].set_xlabel(period)
		plt.savefig("plots/users-dist-distribution.pdf")
		plt.show()

		# histogram of user purchase distances (to investigate their clusterdness)
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			x = spatial.distance.cdist(self.Data[period]["Sales History"], self.Data[period]["Sales History"], metric = "cosine")
			iu1 = np.triu_indices(self.A,1)
			x = x[iu1]
			sns.distplot(x.flatten(),ax=ax[p])
			ax[p].set_xlabel(period)
		plt.savefig("plots/users-dist-distribution.pdf")
		plt.show()


	''' 
		Network based measures of fragmentation
		based on: Kartik Hosanagar, Daniel Fleder, Dokyun Lee, Andreas Buja (2014)
	'''
	def networkAnalysis(self, type = 'preferences'):
		
		AverageDistance = {}
		MedianDegree = {}
		Deviation = {}
		Skewness = {}

		for eng in ["Control"]+self.engine:

			# two types of analysis
			# on the 2d position (preferences, intent) of the users
			if type == "preferences":
				# convert distance matrix to similarity
				matrix = spatial.distance.cdist(self.Data[eng]["Users"], self.Data[eng]["Users"], metric = 'euclidean')
				
			# on the users' purchase history
			if type == "purchases":
				matrix = spatial.distance.cdist(self.Data[eng]["Sales History"], self.Data[eng]["Sales History"], metric = "cosine")
			
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
				AverageDistance.update({eng: AverageDistance["Control"] - np.mean(np.mean(utmatrix))})

			# Average distance
			if eng == "Control":
				Deviation.update({eng: np.std(utmatrix)})
			else:
				Deviation.update({eng: Deviation["Control"] - np.std(utmatrix)})

			# Skewness
			if eng == "Control":
				Skewness.update({eng: stats.skew(utmatrix)})
			else:
				Skewness.update({eng: Skewness["Control"] - stats.skew(utmatrix)})

			# Median degree
			S = []
			for node in nodes:
				S.append(np.sum([G[f][node]["weight"] for f in G.neighbors(node)]))
			if eng == "Control":
				MedianDegree.update({eng: np.median(S)})
			else:
				MedianDegree.update({eng: MedianDegree["Control"] -  np.median(S)})
			
			G.clear()
				
		return {"Average distance":AverageDistance, "Median degree": MedianDegree, "Deviation": Deviation, "Skewness": Skewness }

# Run simulations
d = []
for delta in [10]: # for different types of recommendation salience
	for i in range(1):
		# print("Run:",i)
		
		print("Initialize simulation class...")
		sim = simulation()
		sim.delta = delta
		print("Create simulation instance...")
		sim.createSimulationInstance(seed = i+3)
		sim.simplePlot()
		# time.sleep(10000)
		print("Run simulation...")
		sim.runSimulation()
		print("=== Analysis ===")
		print(" Gini coefficients per rec engine:")
		g = sim.computeGinis()
		print(g)
		print(" Network analytics:")
		n1 = sim.networkAnalysis(type = "purchases")
		print(n1)
		n2 = sim.networkAnalysis(type = "preferences")
		print(n2)
		print("Plotting...")
		sim.plot2D(drift = True) # plotting the user drift is time consuming

		# results in array
		for eng in sim.engine:
			d.append([eng, delta, g[eng], "Gini" ])
			d.append([eng, delta, n1["Average distance"][eng], "Average distance Pu" ])
			d.append([eng, delta, n1["Median degree"][eng], "Median degree Pu" ])
			d.append([eng, delta, n1["Deviation"][eng], "Deviation Pu" ])
			d.append([eng, delta, n1["Skewness"][eng], "Skewness Pu" ])
			d.append([eng, delta, n2["Average distance"][eng], "Average distance Pr" ])
			d.append([eng, delta, n2["Median degree"][eng], "Median degree Pr" ])
			d.append([eng, delta, n2["Deviation"][eng], "Deviation Pr" ])
			d.append([eng, delta, n2["Skewness"][eng], "Skewness Pr" ])

#Store as dataframe
df = pd.DataFrame(d, columns = ["Engine","Delta",'Value',"Metric"])
print(df)
print(df.describe())
df.to_pickle('data.pkl')

df = pd.read_pickle("data.pkl")
g = sns.factorplot(x="Delta", y="Value", hue="Engine", col="Metric", data=df, capsize=.2, palette="YlGnBu_d", size=6, aspect=.75, sharey = False)
g.despine(left=True)
# df.plot.box()
plt.show()


