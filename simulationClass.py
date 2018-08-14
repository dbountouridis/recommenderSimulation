from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2
import random
import time
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
import metrics
#import matplotlib
#matplotlib.use('Agg')
# # force headless backend, or set 'backend' to 'Agg'
# # in your ~/.matplotlib/matplotlibrc
# matplotlib.use('Agg')

# # force non-interactive mode, or set 'interactive' to False
# # in your ~/.matplotlib/matplotlibrc
# matplotlib.pyplot.ioff()


__author__ = 'Dimitrios  Bountouridis'

import bisect
import collections

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

# standardize json output
def standardize(num, precision = 2):
	if precision == 2:
		return float("%.2f"%(num))
	if precision == 4:
		return float("%.4f"%(num))

# export data to plotly json schema
def plotlyjson(x=[],y=[],type_=[],mode="none",title="",ytitle = "",xtitle = ""):
	data = {"x":x,
	"y":y,
	"type":type_,
	"mode":mode}

	layout = {
	"title":title,
	"yaxis":{ "title": ytitle},
	"xaxis":{ "title": xtitle},
	}
	return {"data":data,"layout":layout}

# currently not used
def printj(text, comments=""):
	json = {"action":text,"comments":comments}
	print(json)

# generate initial article prominence
def initialProminceZ0(categories, categoriesSalience, Classes,  plot = False):
	counts = dict(zip(categories, [len(np.where(Classes==i)[0]) for i,c in enumerate(categories) ]))
	items = len(Classes)
	population = categories
	
	# chi square distribution with two degrees of freedom
	df = 2
	mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
	x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), items)
	rv = chi2(df)
	
	Z = {}
	for c in categories: Z.update({c:[]})		
	
	# assign topic to z prominence without replacement
	for i in rv.pdf(x):
		c = choice(population, categoriesSalience)
		while counts[c]<=0:
			c = choice(population, categoriesSalience)
		counts[c]-=1
		Z[c].append(i/0.5)
	if not plot : return Z

	# # plotting
	# min_= np.min([len(Z[i]) for i in Z.keys()])
	# x = []
	# for k in Z.keys():
	# 	x.append(Z[k][:min_])
	# print(np.array(x).T)
	# # set sns context
	# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.2,'text.usetex' : True})
	# sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	# matplotlib.pyplot.rc('text', usetex=True)
	# matplotlib.pyplot.rc('font', family='serif')
	# flatui = sns.color_palette("husl", 8)
	# #fig, ax = plt.subplots()
	# fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(8, 6))
	# ax0= axes
	# cmaps= ['Blues','Reds','Greens','Oranges','Greys']
	# t = ["entertainment","business","sport","politics","tech"]
	# colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(t))]
	# ax0.hist(x, 10, histtype='bar',stacked=True, color=colors,label=categories)
	# ax0.legend(prop={'size': 15})
	# ax0.set_xlabel("$z^0$")
	# ax0.set_ylabel("counts")
	# sns.despine()
	# matplotlib.pyplot.show()

	return Z

class simulation(object):
	''' The main simulation class

		By initiating the class, items and users are generated.
		By changing the recommendation engine, the purchase history and user/items remain the same.

	'''
	def __init__(self):
		# Default settings
		self.seed = 1
		
		# Total number of users
		self.totalNumberOfUsers = 150                        
		self.percentageOfActiveUsersPI = 1.0 
		
		# Amount of distance covered when a user move towards an item 
		self.m = 0.05    

		# Inputs: articles
		self.totalNumberOfIterations = 20
		self.numberOfNewItemsPI = 100 
		
		# Recommendation engines
		self.engine = [] #"max","ItemAttributeKNN","random"]
            		
		# Number of recommended items per iteration per user
		self.n = 5                          
		
		# Choice model
		self.k = 20                          
		self.delta = 5
		self.meanSessionSize = 6                     

		# Awareness, 
		self.theta = 0.07      	# proximity decay
		self.thetaDot = 0.5		# prominence decay
		self.Lambda = 0.6 
		self.V = 40
		
		# slope of salience decrease function
		self.p = 0.1 
		      	        
		self.userVarietySeeking = []
		self.categories = ["entertainment","business","sport","politics","tech"]
		self.categoriesSalience = [0.05,0.07,0.03,0.85,0.01] 
		self.categoriesFrequency = [0.2, 0.2, 0.2, 0.2, 0.2]
		self.AnaylysisInteractionData = []
		self.diversityMetrics = {"EPC": [],"EPCstd": [],'ILD': [],"Gini": [], "EFD": [], "EPD": [], "EILD": [], 'ILDstd': [], "EFDstd": [], "EPDstd": [], "EILDstd": []}
		self.outfolder = ""

	# Create an instance of simulation based on the parameters
	def createSimulationInstance(self):
		random.seed(self.seed)
		np.random.seed(self.seed)

		# compute number of total items in the simulation
		self.I = self.totalNumberOfIterations*self.numberOfNewItemsPI                    
		self.percentageOfActiveItems = self.numberOfNewItemsPI/self.I 

		# GMM on items/articles from the BBC data
		R, S = [5,1,6,7], [5,2,28,28]
		r = int(random.random()*4)
		printj("Item space projection selected:",R[r])
		(X,labels,topicClasses) = pickle.load(open('BBC data/t-SNE-projection'+str(R[r])+'.pkl','rb'))
		gmm = GaussianMixture(n_components=5, random_state=S[r]).fit(X)
		
		# normalize topic weights to sum into 1
		self.categoriesFrequency = [np.round(i,decimals=1) for i in self.categoriesFrequency/np.sum(self.categoriesFrequency)]
		
		# Generate items/articles from the BBC data projection
		self.Items = []
		self.ItemsClass = []
		samples_, classes_ = gmm.sample(self.I*10)
		for c, category in enumerate(self.categories):
			selection = samples_[np.where(classes_ == c)][:int(self.categoriesFrequency[c]*self.I)]
			if len(self.Items) == 0:
				self.Items = np.array(selection)
			else:
				self.Items = np.append(self.Items, selection, axis=0)
			self.ItemsClass+=[c for i in range(len(selection))]
		#samples_, self.ItemsClass = gmm.sample(self.I)
		self.ItemsClass = np.array(self.ItemsClass)
		self.ItemFeatures = gmm.predict_proba(self.Items)
		self.Items = self.Items/55  # scale down to -1, 1 range
		self.ItemDistances = spatial.distance.cdist(self.ItemFeatures, self.ItemFeatures, metric='cosine')

		# generate a random order of item availability
		self.itemOrderOfAppearance = np.arange(self.I).tolist()
		random.shuffle(self.itemOrderOfAppearance)

		# assign initial prominence
		self.categoriesSalience = [np.round(i,decimals=2) for i in self.categoriesSalience/np.sum(self.categoriesSalience)]
		print(self.categoriesSalience)
		Z0 = initialProminceZ0(self.categories, self.categoriesSalience, self.ItemsClass ,plot = True)
		self.initialR = np.zeros(self.I)
		for c, category in enumerate(self.categories): 
			indeces = np.where(self.ItemsClass==c)[0]
			self.initialR[indeces] = Z0[category]

		# dd = spatial.distance.cdist(self.Items, self.Items,metric = 'euclidean')[0]
		# print(np.mean(dd),np.std(dd))
		# fig, ax = matplotlib.pyplot.subplots(2, sharex=True)
		# ax[0].hist(dd, normed=True)
		# matplotlib.pyplot.show()

			

		# Generate users/customers
		# from uniform
		self.Users = np.random.uniform(-1,1,(self.totalNumberOfUsers,2))
		for i, user in enumerate(self.Users):
			while spatial.distance.cdist([user], [[0,0]],metric = 'euclidean')[0][0]>1.1:
				user = np.random.uniform(-1,1,(1,2))[0]
			self.Users[i] = user

		self.UsersClass = [gmm.predict([self.Users[i]*55])[0] for i in range(self.totalNumberOfUsers)]
		#print(self.UsersClass)

	
		# from bivariate
		#self.Users ,_ = make_blobs(n_samples=self.totalNumberOfUsers, n_features=2, centers=1, cluster_std=0.4, center_box=(0, 0), shuffle=True, random_state=self.seed)

		# from circle uniform
		# length = np.random.uniform(0, 1, self.totalNumberOfUsers)
		# angle = np.pi * np.random.uniform(0, 2, self.totalNumberOfUsers)
		# self.Users[:,0] = length * np.cos(angle)
		# self.Users[:,1] = length * np.sin(angle)
		
		
		# Randomly assign how willing each user is to change preferences (used in choice model). 
		# Normal distribution centered around 0.05
		lower, upper = 0, 1
		mu, sigma = 0.1, 0.03
		X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		self.userVarietySeeking = X.rvs(self.totalNumberOfUsers, random_state = self.seed)
		# printj(self.userVarietySeeking)
		# fig, ax = matplotlib.pyplot.subplots(2, sharex=True)
		# ax[0].hist(self.userVarietySeeking, normed=True)
		# matplotlib.pyplot.show()

		# Purchases, sales history
		P = np.zeros([self.totalNumberOfUsers,self.I]) 	
	
		# Create distance matrices
		D = spatial.distance.cdist(self.Users, self.Items,metric = 'euclidean')			
		
		# Store everything in a dictionary structure
		self.Data = {}
		self.Data.update({"ItemProminence" : self.initialR.copy(), 
			"Sales History" : P.copy(),
			"Users" : self.Users.copy(),  
			"D" : D.copy(),
			"ItemLifespan" : np.ones(self.I),
			"Item Has Been Recommended" : np.zeros(self.I),
			"Iterations" : self.totalNumberOfIterations,
			"X" : np.zeros([self.totalNumberOfUsers,self.totalNumberOfIterations]),
			"Y" : np.zeros([self.totalNumberOfUsers,self.totalNumberOfIterations])})
		self.Data["X"][:,0] = self.Data["Users"][:,0]
		self.Data["Y"][:,0] = self.Data["Users"][:,1]

	# export users and items as dataframes
	def exportAnalysisDataAfterIteration(self):
		printj("Exporting per iteration data...", comments = 'Two output pickle files are stored in your workspace.')
		
		# purchase history
		df = pd.DataFrame(self.AnaylysisInteractionData,columns=["Iteration index","User","MML method","Item","Item Age","Item Prominence","Class/Topic","Was Recommended","Agreement between deterministic and stochastic choice", "Item has been recommended before","Class/Topic agreement between deterministic and stochastic choice", "Class/Topic agreement between choice and users main topic","User class","InInitialAwareness"])
		df.to_pickle(self.outfolder + "/dataframe for simple analysis-"+self.engine+".pkl")

		# metrics
		print(self.diversityMetrics)
		df = pd.DataFrame(self.diversityMetrics)
		df["Iteration index"] = np.array([i for i in range(len(self.diversityMetrics["EPC"])) ])
		df["MML method"] = np.array([self.engine for i in  range(len(self.diversityMetrics["EPC"]))])
		df.to_pickle(self.outfolder + "/metrics analysis-"+self.engine+".pkl")

	# if a new recommendation engine is set, then delete the data points so far
	def setEngine(self, engine):
		if self.engine == "Control" and engine!=self.engine:
			self.ControlHistory = self.Data["Sales History"].copy()
		self.engine = engine
		self.AnaylysisInteractionData=[]
		self.diversityMetrics = {"EPC": [],"EPCstd": [],'ILD': [],"Gini": [], "EFD": [], "EPD": [], "EILD": [], 'ILDstd': [], "EFDstd": [], "EPDstd": [], "EILDstd": []}

	# make awareness matrix
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
		#random.seed(self.seed)
		#np.random.seed(self.seed)

		Distances = self.Data["D"][user,:]
		Similarity = -self.k*np.log(Distances)  

		#V = Similarity/np.sum(Similarity)
		V = Similarity.copy()

		if not control: 
			# exponential ranking discount, from Vargas
			for k, r in enumerate(Rec):
				V[r] = Similarity[r] + self.delta*np.power(0.9,k)

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
			dist = spatial.distance.cdist([self.Data["Users"][user]], [self.Items[indexOfChosenItem]],metric = 'euclidean')[0]
			p = np.exp(-(np.power(dist,2))/(self.userVarietySeeking[user])) # based on the awareness formula
			B = np.array(self.Data["Users"][user])
			P = np.array(self.Items[indexOfChosenItem])
			BP = P - B
			x,y = B + self.m*(random.random()<p)*BP 	# probabilistic
			self.Data["Users"][user] = [x,y]
		self.Data["X"][user,iteration] = x
		self.Data["Y"][user,iteration] = y

	# decrease of salience/prominence/prominence
	def prominenceFunction(self, initialProminence, life):
		x = life
		# y = initialProminence*np.power(0.8,x-1)
		# if y<=0.05: y=0
		y = (-self.p*(x-1)+1)*initialProminence
		return max([y, 0])

	# temporal adaptations after the end of each iteration
	def addedFunctionalitiesAfterIteration(self, activeItemIndeces):

		# update user-item distances based on new user positions
		self.Data["D"] = spatial.distance.cdist(self.Data["Users"], self.Items,metric = 'euclidean')	

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
	def exportJsonForOnlineInterface(self, epoch, epoch_index, iterationRange, Awareness, AwarenessOnlyPopular, AwarenessProximity, activeItemIndeces, nonActiveItemIndeces, SalesHistoryBefore):
		Json = {}
		Json.update({"Current recommender" : self.engine})
		Json.update({"Epoch" : epoch})
		Json.update({"Iteration index" : epoch_index})
		Json.update({"Completed" : standardize(epoch_index+1)/len(iterationRange)})
		Json.update({"(median) Number of items in user's awareness" : standardize(np.median(np.sum(Awareness,axis=1)))})
		Json.update({"Number of available, non-available items" : [len(activeItemIndeces),len(nonActiveItemIndeces) ]})
		toProx = np.mean(np.sum(AwarenessProximity[:,activeItemIndeces],axis=1))
		Json.update({"(mean) Number of items in user's awareness due to proximity" : standardize(toProx)})
		toPop = np.mean(np.sum(AwarenessOnlyPopular[:,activeItemIndeces],axis=1))
		Json.update({"(mean) Number of items in user's awareness due to popularity" : standardize(toPop)})
		Json.update({"(mean) Ratio of items in user's awareness due to proximity/popularity" : standardize( toProx/(toPop+toProx))})
		
		ApA = {}
		for i in range(1,10):
			indeces = np.where(self.Data["ItemLifespan"]==i)[0]
			A = Awareness[:,indeces]
			ApA.update({ "Age of "+str(i)+" day(s)" : standardize(np.mean(np.sum(A,axis=1))/np.mean(np.sum(Awareness,axis=1))) })
		Json.update({'Distribution of awareness per article age' : ApA})
		
		f = {}
		for i in range(len(self.categories)):
			indeces = np.where(self.ItemsClass==i)[0]
			A = Awareness[:,indeces]
			f.update({self.categories[i] : standardize(np.mean(np.sum(A,axis=1))/np.mean(np.sum(Awareness,axis=1))) })
		Json.update({"Distribution of awareness per topic" : f})

		#output on terminal
		print(json.dumps(Json, sort_keys=True, indent=4))

		# choice figures
		# Json.update({"Figures":[]})
		f = {}
		values = []
		labels = []
		type_ = "pie"
		title = "Read articles by topic"
		for i in range(len(self.categories)):
			labels.append(self.categories[i])
			indeces = np.where(self.ItemsClass==i)[0]
			A = self.Data["Sales History"][:,indeces]
			f.update({self.categories[i] : standardize(np.sum(np.sum(A,axis=1))/np.sum(np.sum(self.Data["Sales History"],axis=1))) })
			values.append( standardize(np.sum(np.sum(A,axis=1))) )
		Json.update({"Read articles by topic" : f})
		Json.update({"Figure1" : {"values": values,"labels": labels,"type":  type_,"title":title }})

		x = [i for i in range(1,11)]
		y = []
		type_ = "bar"
		title = "Distribution of choice per article age (in days) for the current iteration"
		for i in range(2,12):
			indeces = np.where(self.Data["ItemLifespan"]==i)[0]
			A = self.Data["Sales History"] - SalesHistoryBefore
			A = A[:,indeces]
			y.append( standardize(np.sum(np.sum(A,axis=1))) )
		Json.update({"Figure4" : {"x": x,"y": y,"type": type_,"title": title, "xaxis":{"title": "Article age in days/iterations" }, "yaxis":{"title": "Counts" } }})
		
		# diversity figures
		if self.engine is not "Control":
			x = [i for i in range(len(self.diversityMetrics["ILD"]))]
			y = [standardize(i,precision=4) for i in self.diversityMetrics["ILD"]]
			type_ = "scatter"
			mode = 'lines+markers'
			title = "Unexpectedness diversity"
			Json.update({"Figure2" : {"x": x,"y": y,"type": type_,"title": title, "mode": mode, "xaxis":{"title": "Day" }, "yaxis":{"title": "Unexpectedness diversity" }}})

			x = [i for i in range(len(self.diversityMetrics["EPC"]))]
			y = [standardize(i,precision=4) for i in self.diversityMetrics["EPC"]]
			type_ = "scatter"
			title = "Long-tail diversity"
			Json.update({"Figure3" : {"x": x,"y": y,"type": type_,"title": title,  "mode": mode, "xaxis":{"title": "Day" }, "yaxis":{"title": "Long-tail diversity" }}})
			#self.pickleForMetrics.append([epoch_index,self.engine,met["EPC"],met["ILD"],gini])
	
		# values = []
		# labels = []
		# title = "Distribution of choice per topic"
		# for i in range(len(self.categories)):
		# 	labels.append(self.categories[i])
		# 	indeces = np.where(self.ItemsClass==i)[0]
		# 	A = self.Data["Sales History"][:,indeces]
		# 	values.append( standardize(np.sum(np.sum(A,axis=1))) )
		# Json["Figures"].append( plotlyjson(x = labels, y = values, type_ = "pie" ,mode = "none",title = title, ytitle = "",xtitle = ""))

		# x = [i for i in range(1,11)]
		# y = []
		# type_ = "bar"
		# title = "Distribution of choice per article age (in days) for the current iteration"
		# for i in range(2,12):
		# 	indeces = np.where(self.Data["ItemLifespan"]==i)[0]
		# 	A = self.Data["Sales History"] - SalesHistoryBefore
		# 	A = A[:,indeces]
		# 	y.append( standardize(np.sum(np.sum(A,axis=1))) )
		# Json["Figures"].append( plotlyjson(x = x, y = y, type_ = "bar" ,mode = "none",title = title, ytitle = "Counts",xtitle = "Article age in days/iterations"))
		
		# # diversity figures
		# if self.engine is not "Control":
		# 	x = [i for i in range(len(self.diversityMetrics["ILD"]))]
		# 	y = [standardize(i,precision=4) for i in self.diversityMetrics["ILD"]]
		# 	Json["Figures"].append( plotlyjson(x = x, y = y, type_ = "scatter" ,mode = 'lines+markers',title = "ILD diversity", ytitle = "ILD",xtitle = "Iteration"))

		# 	x = [i for i in range(len(self.diversityMetrics["EPC"]))]
		# 	y = [standardize(i,precision=4) for i in self.diversityMetrics["EPC"]]
		# 	Json["Figures"].append( plotlyjson(x = x, y = y, type_ = "scatter" ,mode = 'lines+markers',title = "EPC diversity", ytitle = "EPC",xtitle = "Iteration"))
		
		# output on file
		Json.update({"Users position" : [(standardize(i[0]),standardize(i[1])) for i in self.Data["Users"]]})
		Json.update({"Items position" : [(standardize(i[0]),standardize(i[1])) for i in self.Items]})
		json.dump(Json, open(self.outfolder + '/'+str(self.engine)+'-data.json', 'w'),sort_keys=True, indent=4)
	
	# run the simulation
	def runSimulation(self, iterationRange =[]):
			
		# for each iteration
		for epoch_index, epoch in enumerate(iterationRange):

			SalesHistoryBefore = self.Data["Sales History"].copy()
			
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

			# only a specific nunber of items in users awareness, minimize the effect of thetas
			for a in range(self.totalNumberOfUsers):
				w = np.where(Awareness[a,:]==1)[0]
				if len(w)>self.V:
					windex = w.tolist()
					random.shuffle(windex)
					Awareness[a,:] = np.zeros(self.I)
					Awareness[a,windex[:self.V]] = 1

			#
			InitialAwareness = Awareness.copy()
	
			# MyMediaLite recommendations 
			if self.engine is not "Control":
				self.exportToMMLdocuments(  activeItemIndeces = activeItemIndeces)
				recommendations = self.mmlRecommendation()
				
			# for each active user
			printj(self.engine+": Choice...")
			for user in activeUserIndeces:
				# if epoch>1: self.simplePlot(forUser = user, awareness = Awareness[user, :], active = activeItemIndeces)
	
				Rec=np.array([-1])
				if self.engine is not "Control":
					if user not in recommendations.keys():
						printj(" -- Nothing to recommend -- to user ",user)
						continue
					Rec = recommendations[user]

					self.Data["Item Has Been Recommended"][Rec] = 1
						
					# temporary adjust awareness for that item-user pair
					Awareness[user, Rec] = 1				

					# if the user has been already purchased the item then decrease awareness of the recommendation
					Awareness[user, np.where(self.Data["Sales History"][user,Rec]>0)[0] ] = 0		

				# select articles
				indecesOfChosenItems,indecesOfChosenItemsW =  self.ChoiceModel(user, Rec, Awareness[user,:], control = self.engine=="Control", sessionSize = int(np.random.normal(self.meanSessionSize, 2)))

				# add item purchase to histories
				self.Data["Sales History"][user, indecesOfChosenItems] += 1		
						
				# compute new user position (we don't store the position in the session, only after it is over)
				
				if self.engine is not "Control":
					self.computeNewPositionOfUserToItem( user, indecesOfChosenItems, epoch)

				# store some data for analysis
				for i,indexOfChosenItem in enumerate(indecesOfChosenItems):
					indexOfChosenItemW = indecesOfChosenItemsW[i]
					self.AnaylysisInteractionData.append([epoch_index, user, self.engine ,indexOfChosenItem,self.Data["ItemLifespan"][indexOfChosenItem], self.Data["ItemProminence"][indexOfChosenItem],self.categories[self.ItemsClass[indexOfChosenItem]],indexOfChosenItem in Rec, indexOfChosenItem == indexOfChosenItemW,self.Data["Item Has Been Recommended"][indexOfChosenItemW],self.ItemsClass[indexOfChosenItem]==self.ItemsClass[indexOfChosenItemW] , self.UsersClass[user]==self.ItemsClass[indexOfChosenItem],self.categories[ self.UsersClass[user]], InitialAwareness[user,indexOfChosenItem] ])

			# after each iteration
			printj(self.engine+": Temporal adaptations...")	
			self.addedFunctionalitiesAfterIteration( activeItemIndeces)

			# compute diversity metrics		
			if self.engine is not "Control":
				met = metrics.metrics(SalesHistoryBefore, recommendations, self.ItemFeatures,self.ItemDistances,self.Data["Sales History"])
				met.update({"Gini": metrics.computeGinis(self.Data["Sales History"],self.ControlHistory)})
				self.diversityMetrics["EPC"].append(met["EPC"])
				self.diversityMetrics["EPCstd"].append(met["EPCstd"])
				self.diversityMetrics["EPD"].append(met["EPD"])
				self.diversityMetrics["EILD"].append(met["EILD"])
				self.diversityMetrics["ILD"].append(met["ILD"])
				self.diversityMetrics["EFD"].append(met["EFD"])
				self.diversityMetrics["EPDstd"].append(met["EPDstd"])
				self.diversityMetrics["EILDstd"].append(met["EILDstd"])
				self.diversityMetrics["ILDstd"].append(met["ILDstd"])
				self.diversityMetrics["EFDstd"].append(met["EFDstd"])
				self.diversityMetrics["Gini"].append(met["Gini"])

			# show stats on screen and save json for interface
			self.exportJsonForOnlineInterface(epoch, epoch_index, iterationRange, Awareness, AwarenessOnlyPopular, AwarenessProximity, activeItemIndeces, nonActiveItemIndeces, SalesHistoryBefore)

		# save results
		self.exportAnalysisDataAfterIteration()
		
	# export to MML type input
	def exportToMMLdocuments(self,  activeItemIndeces = False):
		np.savetxt(self.outfolder + "/users.csv", np.array([i for i in range(self.totalNumberOfUsers)]), delimiter=",", fmt='%d')

		# user profiles
		P = self.Data["Sales History"]

		F = []
		for user in range(P.shape[0]):
			purchases = P[user,:]
			items = np.where(purchases==1)[0]
			userf = self.ItemFeatures[items]
			userfm = np.mean(userf,axis=0)
			userfm = userfm/np.max(userfm)
			feat = np.where(userfm>0.33)[0]
			for f in feat: F.append([int(user),int(f)])
		np.savetxt(self.outfolder + "/users_attributes.csv", np.array(F), delimiter=",", fmt='%d')

		# purchases/positive only feedback
		if activeItemIndeces:
			P = self.Data["Sales History"]
			p = np.where(P>=1)
			z = zip(p[0],p[1])
			l = [[i,j] for i,j in z if j in activeItemIndeces]
			np.savetxt(self.outfolder + "/positive_only_feedback.csv", np.array(l), delimiter=",", fmt='%d')

		# export the active items, or all of them if activeItemIndeces is empty
		if not activeItemIndeces: activeItemIndeces = [i for i in range(self.I)]
		d = []
		for i in activeItemIndeces:
			feat = np.where(self.ItemFeatures[i]/np.max(self.ItemFeatures[i])>0.33)[0]
			for f in feat: d.append([int(i),int(f)])
		np.savetxt(self.outfolder + "/items_attributes.csv", np.array(d), delimiter=",", fmt='%d')
	
	# run MML
	def mmlRecommendation(self):
		# run
		command = "mono MyMediaLite/item_recommendation.exe --training-file=" + self.outfolder + "/positive_only_feedback.csv --item-attributes=" + self.outfolder + "/items_attributes.csv --recommender="+self.engine+" --predict-items-number="+str(self.n)+" --prediction-file=" + self.outfolder + "/output.txt --user-attributes=" + self.outfolder + "/users_attributes.csv --random-seed="+str(int(self.seed*random.random()))
		os.system(command)
		
		# parse output
		f = open( self.outfolder + "/output.txt","r").read() 
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
	
	# plotting	    
	def plot2D(self, drift = False, output = "initial-users-products.pdf", storeOnly = True):

		sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.2})
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		flatui = sns.color_palette("husl", 8)
		f, ax = matplotlib.pyplot.subplots(1,1, figsize=(6,6), sharey=True)

		# products
		cmaps= ['Blues','Reds','Greens','Oranges','Greys']
		colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(self.categories))]
		
		# if no sales history yet, display items with prominence as 3rd dimension
		if np.sum(np.sum(self.Data["Sales History"]))==0:
			n = np.sum(self.Data["Sales History"],axis=0)
			for i in range(self.I): 
				color = colors[self.ItemsClass[i]]
				s = self.Data["ItemProminence"][i]*40
				ax.scatter(self.Items[i,0], self.Items[i,1], marker='o', c=color,s=s,alpha=0.5)
		else:
			# KDE plot
			n = np.sum(self.Data["Sales History"],axis=0)
			for cat in range(len(self.categories)): # 5 topic spaces
				indeces=np.where(self.ItemsClass==cat)[0]
				x = []
				for i in indeces:
					if n[i]>0:
						for k in range(int(n[i])): x.append([self.Items[i,0],self.Items[i,1]])
				ax = sns.kdeplot(np.array(x)[:,0], np.array(x)[:,1], shade=True, shade_lowest=False, alpha = 0.4, cmap=cmaps[cat],kernel='gau')
			
			# scatter
			for i in range(self.I): 
				color = colors[self.ItemsClass[i]]
				if n[i]>=1:
					v = 0.4+ n[i]/np.max(n)*0.4
					c = (1,0,0.0,v)
					s = 2+n[i]/np.max(n)*40
					marker = 'o'
				else:
					color = (0,0,0,0.1)
					v = 0.1
					s = 10
					marker = 'x'
				ax.scatter(self.Items[i,0], self.Items[i,1], marker=marker, c=color,s=s,alpha=v)	
		
		# final user position as a circle
		for i in range(len(self.Users[:,1])):
			ax.scatter(self.Data["Users"][i,0], self.Data["Users"][i,1], marker='D', c='k',s=8, alpha = 0.4 )
		
		# user drift
		if drift:
			for i in range(len(self.Users[:,1])):
				for j in range(len(self.Data["X"][0,:])-1):
					if self.Data["X"][i,j+1]!=0 and self.Data["Y"][i,j+1]!=0:
						ax.plot([self.Data["X"][i,j], self.Data["X"][i,j+1]], [self.Data["Y"][i,j], self.Data["Y"][i,j+1]], 'k-', lw=1, alpha =0.4)

		ax.set_xlabel(self.engine)
		ax.set_aspect('equal', adjustable='box')
		ax.set_xlim([-1.1,1.1])
		ax.set_ylim([-1.1,1.1])
		matplotlib.pyplot.tight_layout()
		matplotlib.pyplot.savefig(self.outfolder + "/" + output)
		if not storeOnly: matplotlib.pyplot.show()

 
def main(argv):
	helpText = 'simulationClass.py  -i <iterationsPerRecommender> -s <seed> -u <totalusers> -d <deltasalience> -r <recommenders> -t <newItemsPerIteration> -f <outfolder> -n <numberOfRecommendations> -p <focusonprominence> -N <meanSessionSize> -w <topicweights> -g <topicprominence>'
	try:
		opts, args = getopt.getopt(argv,"hi:s:u:d:r:t:f:n:p:N:w:g:")
	except getopt.GetoptError:
		printj(helpText)
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			printj(helpText)
			sys.exit()
		elif opt in ("-u"):
			totalNumberOfUsers = int(arg)
		elif opt in ("-i"):
			iterationsPerRecommender = int(arg)
		elif opt in ("-s"):
			seed = int(arg)
		elif opt in ("-d"):
			delta = float(arg)
		elif opt in ("-t"):
			newItemsPerIteration = int(arg)
		elif opt in ("-f"):
			outfolder = arg
		elif opt in ("-n"):
			numberOfRecommendations = int(arg)
		elif opt in ("-N"):
			meanSessionSize = int(arg)
		elif opt in ("-p"):
			focusOnProminentItems = float(arg)
		elif opt in ("-r"):
			if "," in arg: recommenders = arg.split(",") 
			else: recommenders = [arg]
		elif opt in ("-w"):
			if "," in arg: tw = arg.split(",") 
			else: tw = [arg]
			topicweights = np.array([float(i) for i in tw])
		elif opt in ("-g"):
			if "," in arg: tp = arg.split(",") 
			else: tp = [arg]
			topicprominence = np.array([float(i) for i in tp])

	printj("Initialize simulation class...")
	sim = simulation()
	sim.delta, sim.totalNumberOfUsers, sim.numberOfNewItemsPI = delta, totalNumberOfUsers, newItemsPerIteration
	sim.seed = seed
	sim.outfolder = outfolder
	sim.totalNumberOfIterations = iterationsPerRecommender*2 # one for the control and one for each rec
	sim.n = numberOfRecommendations
	sim.Lambda = focusOnProminentItems
	sim.meanSessionSize = meanSessionSize
	sim.categoriesFrequency = topicweights
	sim.categoriesSalience = topicprominence

	printj("Create simulation instance...")
	sim.createSimulationInstance()
	
	# printj("Plotting users/items in 2d space...")
	# sim.plot2D(storeOnly = True)

	printj("Running Control period...", comments = "We first run a Control period without recommenders to deal with the cold start problem.")
	sim.setEngine("Control")
	sim.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender)])
	printj("Saving...", comments = "Output pickle file is stored in your workspace.")
	pickle.dump(sim.Data, open(sim.outfolder + '/Control-data.pkl', 'wb'))
	# printj("Plotting...")
	# sim.plot2D(drift = True, output = "2d-Control.pdf")
	
	printj("Running Recommenders....", comments = "The recommenders continue from the Control period.")
	for rec in recommenders:
		sim2 = copy.deepcopy(sim) 	# continue from the control period
		sim2.setEngine(rec)
		sim2.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender,iterationsPerRecommender*2)])
		printj("Saving for "+rec+"...", comments = "Output pickle file is stored in your workspace.")
		pickle.dump(sim2.Data, open(sim2.outfolder + '/'+rec+'-data.pkl', 'wb'))
		# printj("Plotting for "+rec+"...")
		# sim2.plot2D(drift = True, output = "2d-"+sim2.engine+".pdf")
   
    
if __name__ == "__main__":
   main(sys.argv[1:])           



