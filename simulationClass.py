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
import sys, getopt
import copy
import json


__author__ = 'Dimitrios  Bountouridis'


class simulation(object):
	def __init__(self):
		# Default settings
		self.seed = 1
		
		# Total number of users
		self.totalNumberOfUsers = 150                        
		self.percentageOfActiveUsersPI = 1.0 
		
		# Amount of distance covered when a user move towards an item 
		self.m = 0.05    

		# Inputs: articles
		self.totalNumberOfIterations = 40
		self.numberOfNewItemsPI = 100 
		
		# Recommendation engines
		self.engine = ["BPRMF"] #"max","ItemAttributeKNN","random"]
            		
		# Number of recommended items per iteration per user
		self.n = 5                          
		
		# Choice model
		self.k = 14                          
		self.delta = 10                     

		# Awareness, setting selectes predefined awareness settings
		self.theta = 0.1      
		self.thetaDot = 0.5
		self.Lambda = 0.75 
		
		# slope of salience decrease function
		self.p = 0.1 
		      	        
		self.userVarietySeeking = []
		self.categories = ["entertainment","business","sport","politics","tech"]
		self.categoriesSalience = [0.05,0.07,0.03,0.85,0.01] # arbitrary assigned
		self.pickleForFurtherAnalysis = []

	# Create an instance of simulation based on the parameters
	def createSimulationInstance(self):
		random.seed(self.seed)
		np.random.seed(self.seed)

		# compute number of total items in the simulation
		self.I = self.totalNumberOfIterations*self.numberOfNewItemsPI                    
		self.percentageOfActiveItems = self.numberOfNewItemsPI/self.I 

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

		# Item salience based on topic salience and truncated normal distibution
		self.initialR = np.zeros(self.I)
		#fig, ax = plt.subplots()
		for i in range(len(self.categories)):
			indeces = np.where(self.ItemsClass==i)[0]
			lower, upper = 0, 1
			mu, sigma = self.categoriesSalience[i], 0.1
			a = (lower - mu) / sigma
			b=  (upper - mu) / sigma
			X = stats.truncnorm( a, b, loc=mu, scale=sigma)
			self.initialR[indeces] = X.rvs(len(indeces), random_state = self.seed)
			#x = np.linspace(0,1)
			#ax.plot(x, X.pdf(x)/np.sum(x),'r-', lw=1, alpha=0.6, label=self.categories[i])
		#plt.show()
			

		# Generate users/customers
		# from uniform
		self.Users = np.random.uniform(-1,1,(self.totalNumberOfUsers,2))
		# from bivariate
		#self.Users ,_ = make_blobs(n_samples=self.totalNumberOfUsers, n_features=2, centers=1, cluster_std=0.4, center_box=(0, 0), shuffle=True, random_state=self.seed)
		
		# size of session per user (e.g. amount of articles read per day)
		self.UserSessionSize = [1+int(random.random()*3) for i in range(self.totalNumberOfUsers)]
		
		# Randomly assign how willing each user is to change preferences (used in choice model). 
		# Normal distribution centered around 0.05
		lower, upper = 0, 1
		mu, sigma = 0.05, 0.05
		X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		self.userVarietySeeking = X.rvs(self.totalNumberOfUsers, random_state = self.seed)
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
		self.Data.update({"ItemProminence" : self.initialR.copy(), 
			"Sales History" : P.copy(),
			"Users" : self.Users.copy(),  
			"D" : D.copy(),
			"ItemLifespan" : np.array([1 for i in range(self.I)]),
			"Iterations" : self.totalNumberOfIterations,
			"X" : np.zeros([self.totalNumberOfUsers,self.totalNumberOfIterations]),
			"Y" : np.zeros([self.totalNumberOfUsers,self.totalNumberOfIterations])})
		self.Data["X"][:,0] = self.Data["Users"][:,0]
		self.Data["Y"][:,0] = self.Data["Users"][:,1]

		# Export the user ids once for MML
		self.exportToMMLdocuments(permanent=True)

	# export users and items as dataframes
	def exportToDataframe(self):

		print("Globals statisicts:")
		print(" - Users/readers:")
		UsersDF = pd.DataFrame( list(zip(self.UserSessionSize, self.userVarietySeeking, self.Users[:,0], self.Users[:,1])),columns=["SessionSize","VarietySeeking","X","Y"] )
		print(UsersDF.describe())

		print(" - Items/articles:")
		ItemsDF = pd.DataFrame(self.ItemFeatures, columns = self.categories)
		ItemsDF["Class"] = self.ItemsClass
		ItemsDF["InitialProminence"] = self.initialR
		ItemsDF["X"] = self.Items[:,0]
		ItemsDF["Y"] = self.Items[:,1]
		print(ItemsDF.describe())

		print("Storing dataframes to 'temp'...")
		UsersDF.to_pickle("temp/Users.pkl")
		ItemsDF.to_pickle("temp/Items.pkl")
		#, open('temp/items-features.pkl', 'wb'))

	# if a new recommendation engine is set, then delete the data points so far
	def setEngine(self, engine):
		self.engine = engine
		self.pickleForFurtherAnalysis=[]

	# Make awareness matrix
	def makeawaremx(self):
		random.seed(self.seed)
		np.random.seed(self.seed)

		Dij = self.Data["D"]
		W = np.zeros([self.totalNumberOfUsers,self.I])
		W2 = W.copy() # for analysis purposes
		W3 = W.copy()
		for a in range(self.totalNumberOfUsers):
			for i in range(self.I):
				# W[a,i] = self.Lambda*np.exp(-(1/self.thetaDot)*(np.power(1-self.Data["ItemProminence"][i],2))) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				# W2[a,i] = self.Lambda*np.exp(-(1/self.thetaDot)*  (np.power(1-self.Data["ItemProminence"][i],2))) 
				# W3[a,i] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)

				W[a,i] = self.Lambda*(-self.thetaDot*np.log(1-self.Data["ItemProminence"][i])) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				W2[a,i] = self.Lambda*(-self.thetaDot*np.log(1-self.Data["ItemProminence"][i])) 
				W3[a,i] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				r = random.random()
				W[a,i] = r<W[a,i] # probabilistic
				# W2[a,i] = r<W2[a,i] # probabilistic
				# W3[a,i] = r<W3[a,i] # probabilistic
		return W,W2,W3

	# Probabilistic choice model
	def ChoiceModel(self, user, Rec, w, control = False, sessionSize =1):
		random.seed(self.seed)
		np.random.seed(self.seed)

		Distances = self.Data["D"][user,:]
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
	def computeNewPositionOfUserToItem(self, user, indecesOfChosenItems, iteration):
		random.seed(self.seed)
		np.random.seed(self.seed)
		# compute new user location. But the probability that the user will move towards
		# the item is proportional to their distance
		for indexOfChosenItem in indecesOfChosenItems:
			dist = spatial.distance.cdist([self.Data["Users"][user]], [self.Items[indexOfChosenItem]])[0]
			p = np.exp(-(np.power(dist,2))/(self.userVarietySeeking[user])) # based on the awareness formula
			B = np.array(self.Data["Users"][user])
			P = np.array(self.Items[indexOfChosenItem])
			BP = P - B
			x,y = B + self.m*(random.random()<p)*BP 	# probabilistic
			self.Data["Users"][user] = [x,y]
		self.Data["X"][user,iteration] = x
		self.Data["Y"][user,iteration] = y

	# decrease of salience/prominence/prominence
	def prominenceFunction(self, currentProminence, life):
		x = life
		y = (-self.p*x+1)*currentProminence
		return max([y, 0])

	# temporal adaptations after the end of each iteration
	def addedFunctionalitiesAfterIteration(self, activeItemIndeces):

		# update user-item distances based on new user positions
		self.Data["D"] = spatial.distance.cdist(self.Data["Users"], self.Items)	

		# update lifespan of available items
		self.Data["ItemLifespan"][activeItemIndeces] = self.Data["ItemLifespan"][activeItemIndeces]+1
		
		# update prominence based on lifespan, naive
		for a in activeItemIndeces:
			self.Data['ItemProminence'][a]= self.prominenceFunction(self.initialR[a],self.Data['ItemLifespan'][a])

	# export the subset of available users and items
	def subsetOfAvailableUsersItems(self,iteration):
		
		# user availability
		activeUserIndeces = np.arange(self.totalNumberOfUsers).tolist()
		random.shuffle(activeUserIndeces)
		activeUserIndeces = activeUserIndeces[:int(len(activeUserIndeces)*self.percentageOfActiveUsersPI)] 
		nonActiveUserIndeces = [ i  for i in np.arange(self.totalNumberOfUsers) if i not in activeUserIndeces]

		# items are gradually (at each iteration) becoming available, but have limited lifspan
		activeItemIndeces =[j for j in self.itemOrderOfAppearance[:(iteration+1)*int(self.I*self.percentageOfActiveItems)] if self.Data["ItemProminence"][j]>0]
		nonActiveItemIndeces = [ i  for i in np.arange(self.I) if i not in activeItemIndeces]
		
		return (activeUserIndeces, nonActiveUserIndeces, activeItemIndeces, nonActiveItemIndeces) 

	# export json for online interface
	def exportJsonForOnlineInterface(self, epoch, epoch_index, iterationRange, Awareness, AwarenessOnlyPopular, AwarenessProximity, activeItemIndeces, nonActiveItemIndeces):
		Json = {}
		Json.update({"Current recommender":self.engine})
		Json.update({"Epoch":epoch})
		Json.update({"Iteration index":epoch_index})
		Json.update({"Completed":(epoch_index+1)/len(iterationRange)})
		ApA = {}
		for i in range(1,10):
			indeces = np.where(self.Data["ItemLifespan"]==i)[0]
			A = Awareness[:,indeces]
			ApA.update({ "Age of "+str(i)+" day(s)" : float("%.2f"%(np.mean(np.sum(A,axis=1))/np.mean(np.sum(Awareness,axis=1)))) })
		Json.update({'Distribution of awareness per article age':ApA})
		f = {}
		for i in range(len(self.categories)):
			indeces = np.where(self.ItemsClass==i)[0]
			A = Awareness[:,indeces]
			f.update({self.categories[i]: float("%.2f"%(np.mean(np.sum(A,axis=1))/np.mean(np.sum(Awareness,axis=1)))) })
		Json.update({"Distribution of awareness per topic":f})
		Json.update({"(median) Number of items in user's awareness":np.median(np.sum(Awareness,axis=1))})
		Json.update({"Number of available, non-available items":[len(activeItemIndeces),len(nonActiveItemIndeces) ]})
		Json.update({"(mean) Number of items in user's awareness due to proximity":np.mean(np.sum(AwarenessProximity[:,activeItemIndeces],axis=1))})
		Json.update({"(mean) Number of items in user's awareness due to popularity":np.mean(np.sum(AwarenessOnlyPopular[:,activeItemIndeces],axis=1))})
		print(json.dumps(Json, sort_keys=True, indent=4))
		Json.update({"Users position":[(i[0],i[1]) for i in self.Data["Users"]]})
		Json.update({"Items position":[(i[0],i[1]) for i in self.Items]})
		json.dump(Json, open('temp/'+str(self.engine)+'-data.json', 'w'),sort_keys=True, indent=4)
	

	# Run the simulation
	def runSimulation(self, iterationRange =[]):
			
		# for each iteration
		for epoch_index, epoch in enumerate(iterationRange):
			
			if epoch_index>0: 
				self.Data["Users"][user]  = self.Data["Users"][user].copy()
				self.Data["X"][:,epoch] = self.Data["X"][:,epoch-1]
				self.Data["Y"][:,epoch] = self.Data["Y"][:,epoch-1]
				
			# random subset of available users . Subset of available items to all users
			(activeUserIndeces, nonActiveUserIndeces, activeItemIndeces, nonActiveItemIndeces) = self.subsetOfAvailableUsersItems(epoch)
			
			# compute awareness per user and adjust for availability 
			Awareness, AwarenessOnlyPopular,AwarenessProximity = self.makeawaremx()
			Awareness[:,nonActiveItemIndeces] = 0 

			# do not make available items that a user has purchased before
			Awareness = Awareness - self.Data["Sales History"]>0

			# show stats on screen
			self.exportJsonForOnlineInterface(epoch, epoch_index, iterationRange, Awareness, AwarenessOnlyPopular, AwarenessProximity, activeItemIndeces, nonActiveItemIndeces)
	
			# MyMediaLite recommendations 
			if self.engine is not "Control":
				self.exportToMMLdocuments( permanent = False, activeItemIndeces = activeItemIndeces)

				# recommend using mediaLite
				recommendations = self.mmlRecommendation()			

			# for each active user
			for user in activeUserIndeces:
				# if epoch>1: self.simplePlot(forUser = user, awareness = Awareness[user, :], active = activeItemIndeces)
	
				Rec=np.array([-1])
				if self.engine is not "Control":
					# recommend one of the available items, old version
					#Rec = np.array(activeItemIndeces)[self.recengine(eng, user, activeItemIndeces)] 
					if user not in recommendations.keys():
						print(" -- Nothing to recommend -- to user ",user)
						continue

					Rec = recommendations[user]
						
					# temporary adjust awareness for that item-user pair
					Awareness[user, Rec] = 1				

					# if the user has been already purchased the item then decrease awareness of the recommendation
					Awareness[user, np.where(self.Data["Sales History"][user,Rec]>0)[0] ] = 0		

				# select articles
				indecesOfChosenItems,indecesOfChosenItemsW =  self.ChoiceModel(user, Rec, Awareness[user,:], control = self.engine=="Control", sessionSize = self.UserSessionSize[user])

				# add item purchase to histories
				self.Data["Sales History"][user, indecesOfChosenItems] += 1		
						
				# compute new user position (we don't store the position in the session, only after it is over)
				self.computeNewPositionOfUserToItem( user, indecesOfChosenItems, epoch)

				# store some data for analysis
				for i,indexOfChosenItem in enumerate(indecesOfChosenItems):
					indexOfChosenItemW = indecesOfChosenItemsW[i]
					self.pickleForFurtherAnalysis.append([epoch_index, user, self.engine ,indexOfChosenItem,self.Data["ItemLifespan"][indexOfChosenItem], self.Data["ItemProminence"][indexOfChosenItem],self.ItemsClass[indexOfChosenItem],indexOfChosenItem in Rec, indexOfChosenItem == indexOfChosenItemW ])
					
			# after each iteration	
			self.addedFunctionalitiesAfterIteration( activeItemIndeces)

		# store
		df = pd.DataFrame(self.pickleForFurtherAnalysis,columns=["iteration","userid","engine","itemid","lifespan","inverseSalience","class","wasRecommended","chosenItem_vs_chosenWithoutStochastic"])
		df.to_pickle("temp/history-"+self.engine+".pkl")
		
	# export to MML type input
	def exportToMMLdocuments(self, permanent = True, activeItemIndeces = False):
		np.savetxt("mmlDocuments/users.csv", np.array([i for i in range(self.totalNumberOfUsers)]), delimiter=",", fmt='%d')

		if activeItemIndeces:
			P = self.Data["Sales History"]
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
	def mmlRecommendation(self):
		post = " --random-seed="+str(self.seed)

		# run
		command = "mono mmlDocuments/item_recommendation.exe --training-file=mmlDocuments/positive_only_feedback.csv --item-attributes=mmlDocuments/items_attributes.csv --recommender="+self.engine+"  --predict-items-number="+str(self.n)+" --prediction-file=mmlDocuments/output.txt "+post
		os.system(command)
		
		# parse output
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
	
	# plot initial users, products on 2d plane 
	def simplePlot(self, forUser = False, awareness = False, active = False, storeOnly = True):
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
		if not storeOnly: plt.show()

	# Plotting	    
	def plot2D(self, drift = False, output = "plots/temp.pdf", storeOnly = True):
		#self.Data = pickle.load(open("temp/Data.pkl",'rb'))

		sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		flatui = sns.color_palette("husl", 8)
		f, ax = plt.subplots(1,1, figsize=(6,6), sharey=True)

		# final user position as a circle
		for i in range(len(self.Users[:,1])):
			ax.scatter(self.Data["Users"][i,0], self.Data["Users"][i,1], marker='.', c='b',s=10, alpha = 0.6 )
		
		# user drift
		if drift:
			for i in range(len(self.Users[:,1])):
				for j in range(len(self.Data["X"][0,:])-1):
					if self.Data["X"][i,j+1]!=0 and self.Data["Y"][i,j+1]!=0:
						ax.plot([self.Data["X"][i,j], self.Data["X"][i,j+1]], [self.Data["Y"][i,j], self.Data["Y"][i,j+1]], 'b-', lw=0.5, alpha =0.6)

		# products
		n = np.sum(self.Data["Sales History"],axis=0)
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
			ax.scatter(self.Items[i,0], self.Items[i,1], marker=marker, c=color,s=s,alpha=v)		
		ax.set_xlabel(self.engine)
		ax.set_aspect('equal', adjustable='box')
		plt.tight_layout()
		plt.savefig(output)
		if not storeOnly: plt.show()


# 
def main(argv):
	helpText = 'simulationClass.py  -i <iterationsPerRecommender> -s <seed> -u <totalusers> -d <deltasalience> -r <recommenders> -t <newItemsPerIteration> -f <outfolder>'
	try:
		opts, args = getopt.getopt(argv,"hi:s:u:d:r:t:f:")
	except getopt.GetoptError:
		print(helpText)
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print(helpText)
			sys.exit()
		elif opt in ("-u"):
			totalNumberOfUsers = arg
		elif opt in ("-i"):
			iterationsPerRecommender = int(arg)
		elif opt in ("-s"):
			seed = arg
		elif opt in ("-d"):
			delta = arg
		elif opt in ("-t"):
			newItemsPerIteration = int(arg)
		elif opt in ("-f"):
			outfolder = arg
		elif opt in ("-r"):
			if "," in arg: recommenders = arg.split(",") 
			else: recommenders = [arg]
	
	print("Initialize simulation class...")
	sim = simulation()
	sim.delta = int(delta)
	sim.totalNumberOfUsers = int(totalNumberOfUsers)
	sim.numberOfNewItemsPI = newItemsPerIteration
	sim.seed = int(seed)
	sim.totalNumberOfIterations = iterationsPerRecommender*2 # one for the control and one for each rec

	print("Create simulation instance...")
	sim.createSimulationInstance()
	sim.exportToDataframe()

	print("Plotting users/items in 2d space...")
	sim.simplePlot()

	print("Run Control period...")
	sim.setEngine("Control")
	sim.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender)])
	print("Saving...")
	pickle.dump(sim.Data, open('temp/Control-data.pkl', 'wb'))
	print("Plotting...")
	sim.plot2D(drift = True, output = "plots/2d-Control.pdf")
	
	print("Run Recommenders....")
	for rec in recommenders:
		sim2 = copy.deepcopy(sim) 	# continue from the control period
		sim2.setEngine(rec)
		sim2.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender,iterationsPerRecommender*2)])
		print("Saving for "+rec+"...")
		pickle.dump(sim2.Data, open('temp/'+rec+'-data.pkl', 'wb'))
		print("Plotting for "+rec+"...")
		sim2.plot2D(drift = True, output = "plots/2d-"+sim2.engine+".pdf")
   
    
if __name__ == "__main__":
   main(sys.argv[1:])           



