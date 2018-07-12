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
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import os


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
		
		# Total number of users
		self.totalNumberOfUsers = 150                        
		self.percentageOfActiveUsersPI = 1.0 
		
		# Amount of distance covered when a user move towards an item 
		self.m = 0.05    

		# Inputs: articles
		self.totalNumberOfIterations = 40
		self.numberOfNewItemsPI = 100
		self.I = self.totalNumberOfIterations*self.numberOfNewItemsPI                    
		self.percentageOfActiveItems = self.numberOfNewItemsPI/self.I  
		
		# Recommendation engines
		self.engine = ["BPRMF"] #"max","ItemAttributeKNN","random"]
		
		# Division of iterations per control period vs the rest
		self.iterationSplit = 0.5 
		self.iters1 = [i for i in range(int(self.totalNumberOfIterations*self.iterationSplit))] 
		self.iters2 = [i for i in range(int(self.totalNumberOfIterations*self.iterationSplit),self.totalNumberOfIterations)]              
		
		# Number of recommended items per iteration per user
		self.n = 5                          
		
		# Choice model
		self.k = 14                          
		self.delta = 5                     

		# Awareness, setting selectes predefined awareness settings
		self.theta = 0.1      
		self.thetaDot = 0.5
		self.Lambda = 0.5 
		
		# slope of salience decrease function
		self.p = 0.1 
		      	        
		self.userVarietySeeking = []
		self.categories = ["entertainment","business","sport","politics","tech"]
		self.categoriesSalience = [0.05,0.07,0.03,0.85,0.01] # arbitrary assigned
		self.Pickle = []
			
	# Create an instance of simulation based on the parameters
	def createSimulationInstance(self, seed = None):
		random.seed(seed)

		# Generate items/articles from the BBC data
		(X,labels,topicClasses) = pickle.load(open('BBC data/t-SNE-projection.pkl','rb'))
		gmm = GaussianMixture(n_components=5, random_state=2).fit(X)
		samples_,self.ItemsClass = gmm.sample(self.I)
		self.Items = samples_/55  # scale down to -1, 1 range
		self.ItemFeatures = gmm.predict_proba(samples_)

		# Spearman correlation test between feature representations
		# dist = spatial.distance.cdist(self.Items, self.Items)[0]
		# dist2 = spatial.distance.cdist(self.ItemFeatures, self.ItemFeatures)[0]
		# print(stats.pearsonr(dist,dist2))
		# print(stats.spearmanr(dist,dist2))

		# Fitting a guassian on the pairwise item distances
		# (mu, sigma) = norm.fit(dist)
		# print(mu, sigma)
		# fig, ax = plt.subplots(2, sharex=True)
		# ax[0].hist(dist, normed=True)
		# plt.show()

		# generate a random order of item availability
		self.itemOrderOfAppearance = np.arange(self.I).tolist()
		random.shuffle(self.itemOrderOfAppearance)

		# create initial lifespan for item availability
		L = np.array([1 for i in range(self.I)])

		# Item salience based on topic salience and truncated normal distibution
		self.initialR = np.zeros(self.I)
		#fig, ax = plt.subplots()
		for i in range(5):
			indeces = np.where(self.ItemsClass==i)[0]
			lower, upper = 0, 1
			mu, sigma = self.categoriesSalience[i], 0.1
			a = (lower - mu) / sigma
			b=  (upper - mu) / sigma
			X = stats.truncnorm( a, b, loc=mu, scale=sigma)
			self.initialR[indeces] = X.rvs(len(indeces))
			#x = np.linspace(0,1)
			#ax.plot(x, X.pdf(x)/np.sum(x),'r-', lw=1, alpha=0.6, label=self.categories[i])
		#plt.show()
			

		# Generate users/customers
		# from uniform
		#self.Users = np.random.uniform(-1,1,(self.totalNumberOfUsers,2))
		# from bivariate
		self.Users ,_ = make_blobs(n_samples=self.totalNumberOfUsers, n_features=2, centers=1, cluster_std=0.4, center_box=(0, 0), shuffle=True, random_state=seed)
		
		# size of session per user (e.g. amount of articles read per day)
		self.UserSessionSize = [1+int(random.random()*3) for i in range(self.totalNumberOfUsers)]
		print(self.UserSessionSize)

		# Randomly assign how willing each user is to change preferences (used in choice model). 
		# Normal distribution centered around 0.05
		lower, upper = 0, 1
		mu, sigma = 0.05, 0.05
		X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		self.userVarietySeeking = X.rvs(self.totalNumberOfUsers)
		# print(self.userVarietySeeking)
		# fig, ax = plt.subplots(2, sharex=True)
		# ax[0].hist(self.userVarietySeeking, normed=True)
		# plt.show()

		# Purchases, sales history
		P = np.zeros([self.totalNumberOfUsers,self.I]) 	
	
		# Create distance matrices
		D = spatial.distance.cdist(self.Users, self.Items)			
		
		# Store everything in a dictionary structure
		self.Data = {}
		for eng in self.engine+["Control"]:
			if eng=="Control": iters = self.iters1
			else: iters = self.iters2
			self.Data.update({eng:{"ItemRanking":self.initialR.copy(), "Sales History" : P.copy(),"All Purchased Items" : [],"Users" : self.Users.copy(),"InitialUsers" : self.Users.copy(),  "D" : D.copy(),"ItemLifespan" : L.copy(),"Iterations" : iters,"X" : np.zeros([self.totalNumberOfUsers,len(iters)]),"Y" : np.zeros([self.totalNumberOfUsers,len(iters)])}})
			self.Data[eng]["X"][:,0] = self.Data[eng]["Users"][:,0]
			self.Data[eng]["Y"][:,0] = self.Data[eng]["Users"][:,1]

		# Export the user ids once for MML
		self.exportToMMLdocuments(permanent=True)

	# Make awareness matrix
	def makeawaremx(self,eng):
		random.seed(1)

		Dij = self.Data[eng]["D"]
		W = np.zeros([self.totalNumberOfUsers,self.I])
		W2 = W.copy() # for analysis purposes
		W3 = W.copy()
		for a in range(self.totalNumberOfUsers):
			for i in range(self.I):
				# W[a,i] = self.Lambda*np.exp(-(1/self.thetaDot)*(np.power(1-self.Data[eng]["ItemRanking"][i],2))) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				# W2[a,i] = self.Lambda*np.exp(-(1/self.thetaDot)*  (np.power(1-self.Data[eng]["ItemRanking"][i],2))) 
				# W3[a,i] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)

				W[a,i] = self.Lambda*(-self.thetaDot*np.log(1-self.Data[eng]["ItemRanking"][i])) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				W2[a,i] = self.Lambda*(-self.thetaDot*np.log(1-self.Data[eng]["ItemRanking"][i])) 
				W3[a,i] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				r = random.random()
				W[a,i] = r<W[a,i] # probabilistic
				# W2[a,i] = r<W2[a,i] # probabilistic
				# W3[a,i] = r<W3[a,i] # probabilistic
		return W,W2,W3

	# Probabilistic choice model
	def ChoiceModel(self, eng, user, Rec, w, control = False, sessionSize =1):

		Distances = self.Data[eng]["D"][user,:]
		Similarity = -self.k*np.log(Distances)  

		#V = Similarity/np.sum(Similarity)
		V = Similarity.copy()

		if not control: V[Rec] = 1*Similarity[Rec] + self.delta 

		# Introduce the stochastic component
		E = -np.log(-np.log([random.random() for v in range(len(V))]))
		U = V + E
		sel = np.where(w==1)[0]

		# with stochastic
		selected = np.argsort(U[sel])[::-1]
		
		# without stochastic
		selectedW = np.argsort(V[sel])[::-1]
		return sel[selected[:sessionSize]],sel[selectedW[:sessionSize]]

		randomChoice = False
		if randomChoice:
			sel = np.where(w==1)[0].tolist()
			random.shuffle(sel)
			selected = sel[:sessionSize]
		return selected, selected # index of chosen item

	# Compute new position of a user given they purchased an item
	def computeNewPositionOfUserToItem(self, eng, user, indecesOfChosenItems, iteration):
		# compute new user location. But the probability that the user will move towards
		# the item is proportional to their distance
		for indexOfChosenItem in indecesOfChosenItems:
			dist = spatial.distance.cdist([self.Data[eng]["Users"][user]], [self.Items[indexOfChosenItem]])[0]
			p = np.exp(-(np.power(dist,2))/(self.userVarietySeeking[user])) # based on the awareness formula
			B = np.array(self.Data[eng]["Users"][user])
			P = np.array(self.Items[indexOfChosenItem])
			BP = P - B
			x,y = B + self.m*(random.random()<p)*BP 	# probabilistic
			self.Data[eng]["Users"][user] = [x,y]
		self.Data[eng]["X"][user,iteration] = x
		self.Data[eng]["Y"][user,iteration] = y

	def rankingFunction(self, currentRanking, life):
		x = life
		y = (-self.p*x+1)*currentRanking
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

	def subsetOfAvailableUsersItems(self,iteration, eng):
		
		# user availability
		activeUserIndeces = np.arange(self.totalNumberOfUsers).tolist()
		random.shuffle(activeUserIndeces)
		activeUserIndeces = activeUserIndeces[:int(len(activeUserIndeces)*self.percentageOfActiveUsersPI)] 
		nonActiveUserIndeces = [ i  for i in np.arange(self.totalNumberOfUsers) if i not in activeUserIndeces]

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
					print("    Percentage aware of age",i," :",np.mean(np.sum(A,axis=1))/np.mean(np.sum(Awareness,axis=1)) )
				for i in range(5):
					indeces = np.where(self.ItemsClass==i)[0]
					A = Awareness[:,indeces]
					print("    Percentage aware of class",i," :",np.mean(np.sum(A,axis=1))/np.mean(np.sum(Awareness,axis=1)) )
				print("    Median #aware of:",np.median(np.sum(Awareness,axis=1)))
				print("    Mean aware_score of popularity:",np.mean(np.sum(AwarenessOnlyPopular[:,activeItemIndeces],axis=1)),"Mean aware_score of proximity:",np.mean(np.sum(AwarenessProximity[:,activeItemIndeces],axis=1)))
				print('    Available and non available:',len(activeItemIndeces),len(nonActiveItemIndeces))
				
				# export for medialite
				if eng is not "Control":
					self.exportToMMLdocuments(eng = eng, permanent = False, activeItemIndeces = activeItemIndeces)

					# recommend using mediaLite
					recommendations = self.mmlRecommendation(eng)			

				# for each active user
				for user in activeUserIndeces:
					# if epoch>1: self.simplePlot(forUser = user, awareness = Awareness[user, :], active = activeItemIndeces)

					# the recommendation stays the same for every session per user	
					Rec=np.array([-1])
					if eng is not "Control":
						# recommend one of the available items, old version
						#Rec = np.array(activeItemIndeces)[self.recengine(eng, user, activeItemIndeces)] 
						Rec = recommendations[user]
							
						# temporary adjust awareness for that item-user pair
						Awareness[user, Rec] = 1				

						# if the user has been already purchased the item then decrease awareness of the recommendation
						Awareness[user, np.where(self.Data[eng]["Sales History"][user,Rec]>0)[0] ] = 0		

					# select articles
					indecesOfChosenItems,indecesOfChosenItemsW =  self.ChoiceModel(eng, user, Rec, Awareness[user,:], control = eng=="Control", sessionSize = self.UserSessionSize[user])
					#print(indecesOfChosenItems)

					# add item purchase to histories
					self.Data[eng]["Sales History"][user, indecesOfChosenItems] += 1		
					[self.Data[eng]["All Purchased Items"].append(i) for i in indecesOfChosenItems]		

					# compute new user position (we don't store the position in the session, only after it is over)
					self.computeNewPositionOfUserToItem(eng, user, indecesOfChosenItems, epoch_index)

					# store some data for analysis
					if epoch>9:
						for i,indexOfChosenItem in enumerate(indecesOfChosenItems):
							indexOfChosenItemW = indecesOfChosenItemsW[i]
							self.Pickle.append([epoch_index, user, eng ,indexOfChosenItem,self.Data[eng]["ItemLifespan"][indexOfChosenItem], self.Data[eng]["ItemRanking"][indexOfChosenItem],self.ItemsClass[indexOfChosenItem],indexOfChosenItem in Rec, indexOfChosenItem == indexOfChosenItemW ])
						
				# after each iteration	
				self.addedFunctionalitiesAfterIteration(eng, activeItemIndeces)

		# store
		df = pd.DataFrame(self.Pickle,columns=["iteration","userid","engine","itemid","lifespan","inverseSalience","class","wasRecommended","chosenItem_vs_chosenWithoutStochastic"])
		df.to_pickle("temp/history.pkl")
		pickle.dump(self.Data, open("temp/Data.pkl", "wb"))
		
	# export to MML type input
	def exportToMMLdocuments(self, eng = False, permanent = True, activeItemIndeces = False):

		# Export only the files that remain the same throughout the simulation
		if not eng:
			np.savetxt("mmlDocuments/users.csv", np.array([i for i in range(self.totalNumberOfUsers)]), delimiter=",", fmt='%d')

		if activeItemIndeces:
			P = self.Data[eng]["Sales History"]
			p = np.where(P>=1)
			z = zip(p[0],p[1])
			l = [[i,j] for i,j in z if j in activeItemIndeces]
			np.savetxt("mmlDocuments/positive_only_feedback.csv", np.array(l), delimiter=",", fmt='%d')

		# export the active items, or all of them if activeItemIndeces is empty
		if not activeItemIndeces: activeItemIndeces = [i for i in range(self.I)]
		d = []
		for i in activeItemIndeces:
			feat = np.where(self.ItemFeatures[i]/np.max(self.ItemFeatures[i])>0.33)[0]
			for f in feat: d.append([int(i),int(f)])
		np.savetxt("mmlDocuments/items_attributes.csv", np.array(d), delimiter=",", fmt='%d')
	
	# Run MML
	def mmlRecommendation(self, eng):
		post = ""
		if eng == "max": engine = "MostPopular"
		if eng == "random": engine = "Random"
		if eng == "ItemAttributeKNN": engine = "ItemAttributeKNN"
		if eng == "BPRMF": 
			engine = 'BPRMF'
			#post = "num_factors=10 bias_reg=0 reg_u=0.0025 reg_i=0.0025 reg_j=0.00025 num_iter=30 learn_rate=0.05 uniform_user_sampling=True with_replacement=False update_j=True"

		command = "mono mmlDocuments/item_recommendation.exe --training-file=mmlDocuments/positive_only_feedback.csv --item-attributes=mmlDocuments/items_attributes.csv --recommender="+engine+"  --predict-items-number="+str(self.n)+" --prediction-file=mmlDocuments/output.txt "+post
		os.system(command)
		#print("MML executed")
		f = open("mmlDocuments/output.txt","r").read() 
		f = f.split("\n")
		recommendations = {}
		for line in f[:-1]:
			#if len(line)<1:continue
			l = line.split("\t")
			user_id = int(l[0])
			l1 = l[1].replace("[","").replace("]","").split(",")
			rec = [int(i.split(":")[0]) for i in l1]
			recommendations.update({user_id:rec})
		return recommendations 

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
				Recommendation = np.argsort(np.sum(SubP,axis=0))[::-1][:self.n]

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
				Recommendation = np.argsort(np.sum(SubP,axis=0)/allSales)[::-1][:self.n]
			
		if engine=="min":		# lowest seller
			sales = np.sum(P, axis=0)
			Recommendation = np.argsort(sales)[:self.n]

		if engine=="median":		# median seller
			sales = np.sum(P, axis=0)
			v = np.argsort(sales).tolist()
			Recommendation = v[int(len(v)/2):int(len(v)/2)+self.n]

		if engine=="max":		# highest seller
			sales = np.sum(P, axis=0)
			Recommendation = np.argmax(sales)[:self.n]

		if engine=="top5":		# Top-5 sellers
			sales = np.sum(P, axis=0)
			Recommendation = np.argsort(sales)[::-1][:5]
			
		if engine=="random":		# random
			Recommendation = [int(random.random()*P.shape[1]) for i in range(self.n)]

		return Recommendation

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
			for i in range(self.I):
				color = flatui[self.ItemsClass[i]]
				ax.scatter(self.Items[i,0], self.Items[i,1], marker='o', c=color,s=20*self.initialR[i], alpha =0.8)	
			for i in range(self.totalNumberOfUsers):
				ax.scatter(self.Users[i,0], self.Users[i,1], marker='+', c='b',s=40,alpha=1,edgecolors="k",linewidths=0.5)
		else:
			#print(np.where(np.array(awareness)==1)[0])
			for i in np.where(np.array(awareness)==1)[0]:
				color = flatui[self.ItemsClass[i]]
				ax.scatter(self.Items[i,0], self.Items[i,1], marker='o', c=color,s=20*self.initialR[i], alpha =1)	
			for i in np.where(np.array(awareness)==0)[0]:
				if i in active:
					ax.scatter(self.Items[i,0], self.Items[i,1], marker='o', c='k',s=10*self.initialR[i], alpha =0.1)
			ax.scatter(self.Users[forUser,0], self.Users[forUser,1], marker='+', c='b',s=40,alpha=self.userVarietySeeking[forUser],edgecolors="k",linewidths=0.5)	

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
			iu1 = np.triu_indices(self.totalNumberOfUsers,1)
			x = x[iu1]
			sns.distplot(x.flatten(),ax=ax[p])
			ax[p].set_xlabel(period)
		plt.savefig("plots/users-dist-distribution.pdf")
		plt.show()

		# histogram of user purchase distances (to investigate their clusterdness)
		f, ax = plt.subplots(1,1+len(self.engine), figsize=(15,6), sharey=True)
		for p, period in enumerate(["Control"]+self.engine):
			x = spatial.distance.cdist(self.Data[period]["Sales History"], self.Data[period]["Sales History"], metric = "cosine")
			iu1 = np.triu_indices(self.totalNumberOfUsers,1)
			x = x[iu1]
			sns.distplot(x.flatten(),ax=ax[p])
			ax[p].set_xlabel(period)
		plt.savefig("plots/users-dist-distribution.pdf")
		plt.show()

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


# 
def main(argv):
    
    print("Initialize simulation class...")
	sim = simulation()
	sim.delta = delta
	print("Create simulation instance...")
	sim.createSimulationInstance(seed = i+3)
	#sim.simplePlot()
	print("Run simulation...")
	sim.runSimulation()
	#print("Plotting...")
	#sim.plot2D(drift = True) # plotting the user drift is time consuming
   
    
if __name__ == "__main__":
   main(sys.argv[1:])           



