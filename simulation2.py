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
import matplotlib
import bisect
import collections
#matplotlib.use('Agg')
# # force headless backend, or set 'backend' to 'Agg'
# # in your ~/.matplotlib/matplotlibrc
# matplotlib.use('Agg')

# # force non-interactive mode, or set 'interactive' to False
# # in your ~/.matplotlib/matplotlibrc
# matplotlib.pyplot.ioff()


__author__ = 'Dimitrios  Bountouridis'

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
	#random.seed(seed)
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


class users(object):
	def __init__(self):
		self.seed = 1

		# Total number of users
		self.totalNumberOfUsers = 150                        
		self.percentageOfActiveUsersPI = 1.0 

		# Amount of distance covered when a user move towards an item 
		self.m = 0.05  

		# Choice model
		self.k = 20                          
		self.delta = 5
		self.beta = 0.9     
		self.meanSessionSize = 6                     

		# Awareness, 
		self.theta = 0.07      	# proximity decay
		self.thetaDot = 0.5		# prominence decay
		self.Lambda = 0.6 
		self.w = 40 			# maximum awareness pool size (w in paper)
		self.Awareness = False

		# 
		self.Users = False
		self.UsersClass = False
		self.userVarietySeeking = False
		self.X = False
		self.Y = False

	def generatePopulation(self):
		random.seed(self.seed)
		np.random.seed(self.seed)
		
		# Position on the topic space
		self.Users = np.random.uniform(-1,1,(self.totalNumberOfUsers,2))
		for i, user in enumerate(self.Users):
			while spatial.distance.cdist([user], [[0,0]],metric = 'euclidean')[0][0]>1.1:
				user = np.random.uniform(-1,1,(1,2))[0]
			self.Users[i] = user
	
		# Variety seeking, willingness to drift
		lower, upper = 0, 1
		mu, sigma = 0.1, 0.03
		X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		self.userVarietySeeking = X.rvs(self.totalNumberOfUsers, random_state = self.seed)

		#self.UsersClass = [gmm.predict([self.Users[i]*55])[0] for i in range(self.totalNumberOfUsers)]

		self.X = {i:[self.Users[i,0]] for i in range(self.totalNumberOfUsers)}
		self.Y = {i:[self.Users[i,1]] for i in range(self.totalNumberOfUsers)}

	# draw the session size of each user at each iteration from a normal distribution
	def sessionSize(self):
		return int(np.random.normal(self.meanSessionSize, 2))

	# randomly select a subset of the users (not used in the experiments)
	def subsetOfAvailableUsers(self):
		# user availability
		self.activeUserIndeces = np.arange(self.totalNumberOfUsers).tolist()
		random.shuffle(self.activeUserIndeces)
		self.activeUserIndeces = self.activeUserIndeces[:int(len(self.activeUserIndeces)*self.percentageOfActiveUsersPI)] 
		self.nonActiveUserIndeces = [ i  for i in np.arange(self.totalNumberOfUsers) if i not in self.activeUserIndeces]

	# compute awareness from proximity and prominence (not considering availability, recommendations, history)
	def computeAwarenessMatrix(self, Dij, ItemProminence):
		totalNumberOfItems = ItemProminence.shape[0]

		W = np.zeros([self.totalNumberOfUsers,totalNumberOfItems])
		W2 = W.copy() # for analysis purposes
		W3 = W.copy()
		for a in range(self.totalNumberOfUsers):
			for i in range(totalNumberOfItems):

				W[a,i] = self.Lambda*(-self.thetaDot*np.log(1-ItemProminence[i])) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				W2[a,i] = self.Lambda*(-self.thetaDot*np.log(1-ItemProminence[i])) 
				W3[a,i] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,i],2))/self.theta)
				r = random.random()
				W[a,i] = r<W[a,i] # probabilistic	
		self.Awareness, self.AwarenessOnlyPopular, self.AwarenessProximity =  W, W2, W3

	# select items a user
	def ChoiceModule(self, Rec, w, distanceToItems, sessionSize, control = False):

		Similarity = -self.k*np.log(distanceToItems)  
		V = Similarity.copy()

		if not control: 
			# exponential ranking discount, from Vargas
			for k, r in enumerate(Rec):
				V[r] = Similarity[r] + self.delta*np.power(self.beta,k)

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

	# Compute new position of a user given their purchased item(s)
	def computeNewPositionOfUser(self, user, ChosenItems):
		for itemPosition in ChosenItems:
			dist = spatial.distance.cdist([self.Users[user]], [itemPosition],metric = 'euclidean')[0]
			p = np.exp(-(np.power(dist,2))/(self.userVarietySeeking[user])) # based on the awareness formula
			B = np.array(self.Users[user])
			P = np.array(itemPosition)
			BP = P - B
			x,y = B + self.m*(random.random()<p)*BP 	# probabilistic
			self.Users[user] = [x,y]
		self.X[user].append(x)
		self.Y[user].append(y)

	def showSettings(self):
		variables = [(key,type(self.__dict__[key])) for key in self.__dict__.keys() if (type(self.__dict__[key]) is str or type(self.__dict__[key]) is float or type(self.__dict__[key]) is int )]

		Json={ key: self.__dict__[key] for key,tp in variables }
		print(json.dumps(Json, sort_keys=True, indent=4))


class items(object):
	def __init__(self):
		self.seed = 1
		self.numberOfNewItemsPI = 100 

		# Number of recommended items per iteration per user
		self.n = 5 
                    		
		# slope of salience decrease function
		self.p = 0.1 
		
		# default topics and weights   	        
		self.topics = ["entertainment","business","sport","politics","tech"]
		self.topicsProminence = [0.05,0.07,0.03,0.85,0.01] 
		self.topicsFrequency = [0.2, 0.2, 0.2, 0.2, 0.2]

		self.totalNumberOfItems = False
		self.percentageOfActiveItems = False

		self.Items = []
		self.ItemsClass = []
		self.ItemsFeatures = False
		self.ItemsDistances = False
		self.ItemsOrderOfAppearance = False
		self.ItemProminence = False
		self.ItemLifespan = False
		self.hasBeenRecommended = False

	def generatePopulation(self, totalNumberOfIterations):
		random.seed(self.seed)
		np.random.seed(self.seed)

				# compute number of total items in the simulation
		self.totalNumberOfItems = totalNumberOfIterations*self.numberOfNewItemsPI                    
		self.percentageOfActiveItems = self.numberOfNewItemsPI/self.totalNumberOfItems

		# GMM on items/articles from the BBC data
		R, S = [5,1,6,7], [5,2,28,28]
		r = int(random.random()*4)
		printj("Item space projection selected:",R[r])
		(X,labels,topicClasses) = pickle.load(open('BBC data/t-SNE-projection'+str(R[r])+'.pkl','rb'))
		gmm = GaussianMixture(n_components=5, random_state=S[r]).fit(X)
		
		# normalize topic weights to sum into 1 (CBF)
		self.topicsFrequency = [np.round(i,decimals=1) for i in self.topicsFrequency/np.sum(self.topicsFrequency)]
		
		# Generate items/articles from the BBC data projection
		samples_, classes_ = gmm.sample(self.totalNumberOfItems*10)
		for c, category in enumerate(self.topics):
			selection = samples_[np.where(classes_ == c)][:int(self.topicsFrequency[c]*self.totalNumberOfItems)]
			if len(self.Items) == 0:
				self.Items = np.array(selection)
			else:
				self.Items = np.append(self.Items, selection, axis=0)
			self.ItemsClass+=[c for i in range(len(selection))]
		#samples_, self.ItemsClass = gmm.sample(self.totalNumberOfItems)
		self.ItemsClass = np.array(self.ItemsClass)
		self.ItemsFeatures = gmm.predict_proba(self.Items)
		self.Items = self.Items/55  # scale down to -1, 1 range
		
		# cosine distance between item features
		self.ItemsDistances = spatial.distance.cdist(self.ItemsFeatures, self.ItemsFeatures, metric='cosine')

		# generate a random order of item availability
		self.ItemsOrderOfAppearance = np.arange(self.totalNumberOfItems).tolist()
		random.shuffle(self.ItemsOrderOfAppearance)
	
		# initial prominence
		self.initialProminceZ0()
		self.ItemProminence = self.ItemsInitialProminence.copy()

		# lifespan
		self.ItemLifespan = np.ones(self.totalNumberOfItems)

		# has been recommended before
		self.hasBeenRecommended = np.zeros(self.totalNumberOfItems)

	# decrease of salience/prominence/prominence
	def prominenceFunction(self, initialProminence, life):
		x = life
		# y = initialProminence*np.power(0.8,x-1)
		# if y<=0.05: y=0
		y = (-self.p*(x-1)+1)*initialProminence
		return max([y, 0])

	def subsetOfAvailableItems(self,iteration):
		
		# items are gradually (at each iteration) becoming available, but have limited lifspan
		self.activeItemIndeces =[j for j in self.ItemsOrderOfAppearance[:(iteration+1)*int(self.totalNumberOfItems*self.percentageOfActiveItems)] if self.ItemProminence[j]>0]
		self.nonActiveItemIndeces = [ i  for i in np.arange(self.totalNumberOfItems) if i not in self.activeItemIndeces]

	def updateProminence(self):
		# update lifespan of available items
		self.ItemLifespan[self.activeItemIndeces] = self.ItemLifespan[self.activeItemIndeces]+1
		
		# update prominence based on lifespan, naive
		for a in self.activeItemIndeces:
			self.ItemProminence[a] = self.prominenceFunction(self.ItemsInitialProminence[a],self.ItemLifespan[a])
		
	# generate initial article prominence
	def initialProminceZ0(self):

		# CBF
		self.topicsProminence = [np.round(i,decimals=2) for i in self.topicsProminence/np.sum(self.topicsProminence)]

		counts = dict(zip(self.topics, [len(np.where(self.ItemsClass==i)[0]) for i,c in enumerate(self.topics) ]))
		items = len(self.ItemsClass)
		population = self.topics
		
		# chi square distribution with two degrees of freedom
		df = 2
		mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
		x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), items)
		rv = chi2(df)
		
		Z = {}
		for c in self.topics: Z.update({c:[]})		
		
		# assign topic to z prominence without replacement
		for i in rv.pdf(x):
			c = choice(population, self.topicsProminence)
			while counts[c]<=0:
				c = choice(population, self.topicsProminence)
			counts[c]-=1
			Z[c].append(i/0.5)

		self.ItemsInitialProminence = np.zeros(self.totalNumberOfItems)
		for c, category in enumerate(self.topics): 
			indeces = np.where(self.ItemsClass==c)[0]
			self.ItemsInitialProminence[indeces] = Z[category]	

		# # plotting
		# min_= np.min([len(Z[i]) for i in Z.keys()])
		# x = []
		# for k in Z.keys():
		# 	x.append(Z[k][:min_])
		# print(np.array(x).T)
		# # set sns context
		# sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1.0,'xtick.labelsize': 32, 'axes.labelsize': 32})
		# sns.set(style="whitegrid")
		# sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		# matplotlib.pyplot.rc('text', usetex=True)
		# matplotlib.pyplot.rc('font', family='serif',size=20)
		# flatui = sns.color_palette("husl", 8)
		# #fig, ax = plt.subplots()
		# fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(8, 6))
		# ax0= axes
		# cmaps= ['Blues','Reds','Greens','Oranges','Greys']
		# t = ["entertainment","business","sport","politics","tech"]
		# colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(t))]
		# ax0.hist(x, 10, histtype='bar',stacked=True, color=colors,label=categories)
		# ax0.legend(prop={'size': 18})
		# for tick in ax0.xaxis.get_major_ticks():
		# 	tick.label.set_fontsize(18)
		# for tick in ax0.yaxis.get_major_ticks():
		# 	tick.label.set_fontsize(0)
		# ax0.set_xlabel("$z^0$",fontsize=20)
		# ax0.set_ylabel("")
		# sns.despine()
		# matplotlib.pyplot.show()

	def showSettings(self):
		variables = [key for key in self.__dict__.keys() if (type(self.__dict__[key]) is str or type(self.__dict__[key]) is float or type(self.__dict__[key]) is int or type(self.__dict__[key]) is list and len(self.__dict__[key])<10)]
		old = self.__dict__
		Json={ key: old[key] for key in variables }
		print(json.dumps(Json, sort_keys=True, indent=4))


class simulation(object):
	def __init__(self):
		# Default settings
		
		self.totalNumberOfIterations = 20
		self.algorithm = "Control"	
		self.AnaylysisInteractionData = []
		self.diversityMetrics = {"EPC": [],"EPCstd": [],'ILD': [],"Gini": [], "EFD": [], "EPD": [], "EILD": [], 'ILDstd': [], "EFDstd": [], "EPDstd": [], "EILDstd": []}
		self.outfolder = ""

		#self.U = False
		#self.I = False
	# Create an instance of simulation based on the parameters
	def createSimulationInstance(self):
		
		# Distance matrix between users and items
		self.D = spatial.distance.cdist(self.U.Users, self.I.Items, metric = 'euclidean')

		self.SalesHistory = np.zeros([self.U.totalNumberOfUsers,self.I.totalNumberOfItems]) 			
				
	# export users and items as dataframes
	def exportAnalysisDataAfterIteration(self):
		printj("Exporting per iteration data...", comments = 'Two output pickle files are stored in your workspace.')
		
		# purchase history
		df = pd.DataFrame(self.AnaylysisInteractionData,columns=["Iteration index","User","MML method","Item","Item Age","Item Prominence","Class/Topic","Was Recommended","Agreement between deterministic and stochastic choice", "Item has been recommended before","Class/Topic agreement between deterministic and stochastic choice", "Class/Topic agreement between choice and users main topic","User class","InInitialAwareness"])
		df.to_pickle(self.outfolder + "/dataframe for simple analysis-"+self.algorithm+".pkl")

		# metrics
		#print(self.diversityMetrics)
		df = pd.DataFrame(self.diversityMetrics)
		df["Iteration index"] = np.array([i for i in range(len(self.diversityMetrics["EPC"])) ])
		df["MML method"] = np.array([self.algorithm for i in  range(len(self.diversityMetrics["EPC"]))])
		df.to_pickle(self.outfolder + "/metrics analysis-"+self.algorithm+".pkl")

	# if a new recommendation algorithm is set, then delete the data points so far
	def setAlgorithm(self, engine):
		if self.algorithm == "Control" and engine!=self.algorithm:
			self.ControlHistory = self.SalesHistory.copy()
		self.algorithm = engine
		self.AnaylysisInteractionData=[]
		self.diversityMetrics = {"EPC": [],"EPCstd": [],'ILD': [],"Gini": [], "EFD": [], "EPD": [], "EILD": [], 'ILDstd': [], "EFDstd": [], "EPDstd": [], "EILDstd": []}

	# export json for online interface
	def exportJsonForOnlineInterface(self, epoch, epoch_index, iterationRange, SalesHistoryBefore):
		Json = {}
		Json.update({"Current recommender" : self.algorithm})
		Json.update({"Epoch" : epoch})
		Json.update({"Iteration index" : epoch_index})
		Json.update({"Completed" : standardize(epoch_index+1)/len(iterationRange)})
		Json.update({"(median) Number of items in user's awareness" : standardize(np.median(np.sum(self.U.Awareness,axis=1)))})
		Json.update({"Number of available, non-available items" : [len(self.I.activeItemIndeces),len(self.I.nonActiveItemIndeces) ]})
		toProx = np.mean(np.sum(self.U.AwarenessProximity[:,self.I.activeItemIndeces],axis=1))
		Json.update({"(mean) Number of items in user's awareness due to proximity" : standardize(toProx)})
		toPop = np.mean(np.sum(self.U.AwarenessOnlyPopular[:,self.I.activeItemIndeces],axis=1))
		Json.update({"(mean) Number of items in user's awareness due to popularity" : standardize(toPop)})
		Json.update({"(mean) Ratio of items in user's awareness due to proximity/popularity" : standardize( toProx/(toPop+toProx))})
		
		ApA = {}
		for i in range(1,10):
			indeces = np.where(self.I.ItemLifespan==i)[0]
			A = self.U.Awareness[:,indeces]
			ApA.update({ "Age of "+str(i)+" day(s)" : standardize(np.mean(np.sum(A,axis=1))/np.mean(np.sum(self.U.Awareness,axis=1))) })
		Json.update({'Distribution of awareness per article age' : ApA})
		
		f = {}
		for i in range(len(self.I.topics)):
			indeces = np.where(self.I.ItemsClass==i)[0]
			A = self.U.Awareness[:,indeces]
			f.update({self.I.topics[i] : standardize(np.mean(np.sum(A,axis=1))/np.mean(np.sum(self.U.Awareness,axis=1))) })
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
		for i in range(len(self.I.topics)):
			labels.append(self.I.topics[i])
			indeces = np.where(self.I.ItemsClass==i)[0]
			A = self.SalesHistory[:,indeces]
			f.update({self.I.topics[i] : standardize(np.sum(np.sum(A,axis=1))/np.sum(np.sum(self.SalesHistory,axis=1))) })
			values.append( standardize(np.sum(np.sum(A,axis=1))) )
		Json.update({"Read articles by topic" : f})
		Json.update({"Figure1" : {"values": values,"labels": labels,"type":  type_,"title":title }})

		x = [i for i in range(1,11)]
		y = []
		type_ = "bar"
		title = "Distribution of choice per article age (in days) for the current iteration"
		for i in range(2,12):
			indeces = np.where(self.I.ItemLifespan==i)[0]
			A = self.SalesHistory - SalesHistoryBefore
			A = A[:,indeces]
			y.append( standardize(np.sum(np.sum(A,axis=1))) )
		Json.update({"Figure4" : {"x": x,"y": y,"type": type_,"title": title, "xaxis":{"title": "Article age in days/iterations" }, "yaxis":{"title": "Counts" } }})
		
		# diversity figures
		if self.algorithm is not "Control":
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
			#self.pickleForMetrics.append([epoch_index,self.algorithm,met["EPC"],met["ILD"],gini])
	
		# values = []
		# labels = []
		# title = "Distribution of choice per topic"
		# for i in range(len(self.topics)):
		# 	labels.append(self.topics[i])
		# 	indeces = np.where(self.ItemsClass==i)[0]
		# 	A = self.SalesHistory[:,indeces]
		# 	values.append( standardize(np.sum(np.sum(A,axis=1))) )
		# Json["Figures"].append( plotlyjson(x = labels, y = values, type_ = "pie" ,mode = "none",title = title, ytitle = "",xtitle = ""))

		# x = [i for i in range(1,11)]
		# y = []
		# type_ = "bar"
		# title = "Distribution of choice per article age (in days) for the current iteration"
		# for i in range(2,12):
		# 	indeces = np.where(self.I.ItemLifespan==i)[0]
		# 	A = self.SalesHistory - SalesHistoryBefore
		# 	A = A[:,indeces]
		# 	y.append( standardize(np.sum(np.sum(A,axis=1))) )
		# Json["Figures"].append( plotlyjson(x = x, y = y, type_ = "bar" ,mode = "none",title = title, ytitle = "Counts",xtitle = "Article age in days/iterations"))
		
		# # diversity figures
		# if self.algorithm is not "Control":
		# 	x = [i for i in range(len(self.diversityMetrics["ILD"]))]
		# 	y = [standardize(i,precision=4) for i in self.diversityMetrics["ILD"]]
		# 	Json["Figures"].append( plotlyjson(x = x, y = y, type_ = "scatter" ,mode = 'lines+markers',title = "ILD diversity", ytitle = "ILD",xtitle = "Iteration"))

		# 	x = [i for i in range(len(self.diversityMetrics["EPC"]))]
		# 	y = [standardize(i,precision=4) for i in self.diversityMetrics["EPC"]]
		# 	Json["Figures"].append( plotlyjson(x = x, y = y, type_ = "scatter" ,mode = 'lines+markers',title = "EPC diversity", ytitle = "EPC",xtitle = "Iteration"))
		
		# output on file
		Json.update({"Users position" : [(standardize(i[0]),standardize(i[1])) for i in self.U.Users]})
		Json.update({"Items position" : [(standardize(i[0]),standardize(i[1])) for i in self.I.Items]})
		json.dump(Json, open(self.outfolder + '/'+str(self.algorithm)+'-data.json', 'w'),sort_keys=True, indent=4)
	
	def AwarenessModule(self, epoch):
		# random subset of available users . Subset of available items to all users
		self.U.subsetOfAvailableUsers()
		self.I.subsetOfAvailableItems(epoch)
		
		# compute initial awareness per user
		self.U.computeAwarenessMatrix(self.D, self.I.ItemProminence)
		
		# adjust for availability 
		self.U.Awareness[:,self.I.nonActiveItemIndeces] = 0 

		# do not make available items that a user has purchased before
		self.U.Awareness = self.U.Awareness - self.SalesHistory>0

		# only a specific nunber of items in users awareness, minimize the effect of thetas
		for a in range(self.U.totalNumberOfUsers):
			w = np.where(self.U.Awareness[a,:]==1)[0]
			if len(w)>self.U.w:
				windex = w.tolist()
				random.shuffle(windex)
				self.U.Awareness[a,:] = np.zeros(self.I.totalNumberOfItems)
				self.U.Awareness[a,windex[:self.U.w]] = 1
		
	def TemporalAdaptationsModule(self):
		
		# update user-item distances based on new user positions
		if self.algorithm is not "Control":
			self.D = spatial.distance.cdist(self.U.Users, self.I.Items, metric = 'euclidean')	

		# update the items' prominence
		self.I.updateProminence()

	# run the simulation
	def runSimulation(self, iterationRange =[]):
			
		# for each iteration
		for epoch_index, epoch in enumerate(iterationRange):

			# initializations prior to the iteration
			SalesHistoryBefore = self.SalesHistory.copy()
				
			# Awareness from proximity and prominence
			self.AwarenessModule(epoch)
			InitialAwareness = self.U.Awareness.copy()
		
			# Recommendation module 
			if self.algorithm is not "Control":
				self.exportToMMLdocuments()
				recommendations = self.mmlRecommendation()
	
			# Add recommendations to each user's awareness pool			
			for user in self.U.activeUserIndeces:
				Rec=np.array([-1])
				
				if self.algorithm is not "Control":
					if user not in recommendations.keys():
						printj(" -- Nothing to recommend -- to user ",user)
						continue
					Rec = recommendations[user]
					self.I.hasBeenRecommended[Rec] = 1
						
					# temporary adjust awareness for that item-user pair
					self.U.Awareness[user, Rec] = 1				

					# if the user has been already purchased the item then decrease awareness of the recommendation
					self.U.Awareness[user, np.where(self.SalesHistory[user,Rec]>0)[0] ] = 0		

			# Choice module
			for user in self.U.activeUserIndeces:
				Rec=np.array([-1])
				
				if self.algorithm is not "Control":
					if user not in recommendations.keys():
						printj(" -- Nothing to recommend -- to user ",user)
						continue
					Rec = recommendations[user]
				# select articles
				indecesOfChosenItems,indecesOfChosenItemsW =  self.U.ChoiceModule(Rec, self.U.Awareness[user,:], self.D[user,:], self.U.sessionSize(), control = self.algorithm=="Control")

				# add item purchase to histories
				self.SalesHistory[user, indecesOfChosenItems] += 1		
						
				# compute new user position (we don't update the position yet, only after the iteration is over)
				if self.algorithm is not "Control" and len(indecesOfChosenItems)>0:
					self.U.computeNewPositionOfUser(user, self.I.Items[indecesOfChosenItems])

				# store some data for analysis
				for i,indexOfChosenItem in enumerate(indecesOfChosenItems):
					indexOfChosenItemW = indecesOfChosenItemsW[i]
					self.AnaylysisInteractionData.append([epoch_index, user, 
						self.algorithm ,indexOfChosenItem,self.I.ItemLifespan[indexOfChosenItem], 
						self.I.ItemProminence[indexOfChosenItem],
						self.I.topics[self.I.ItemsClass[indexOfChosenItem]],indexOfChosenItem in Rec, 
						indexOfChosenItem == indexOfChosenItemW,self.I.hasBeenRecommended[indexOfChosenItemW],self.I.ItemsClass[indexOfChosenItem]==self.I.ItemsClass[indexOfChosenItemW] , 
						0,
						0, 
						InitialAwareness[user,indexOfChosenItem] ])

			# Temporal adaptations
			printj(self.algorithm+": Temporal adaptations...")	
			self.TemporalAdaptationsModule()

			# compute diversity metrics		
			if self.algorithm is not "Control":
				met = metrics.metrics(SalesHistoryBefore, recommendations, self.I.ItemsFeatures, self.I.ItemsDistances, self.SalesHistory)
				met.update({"Gini": metrics.computeGinis(self.SalesHistory,self.ControlHistory)})
				for key in met.keys():
					self.diversityMetrics[key].append(met[key])

			# show stats on screen and save json for interface
			self.exportJsonForOnlineInterface(epoch, epoch_index, iterationRange, SalesHistoryBefore)

		# save results
		self.exportAnalysisDataAfterIteration()
		
	# export to MML type input
	def exportToMMLdocuments(self):
		np.savetxt(self.outfolder + "/users.csv", np.array([i for i in range(self.U.totalNumberOfUsers)]), delimiter=",", fmt='%d')

		F = []
		for user in range(self.SalesHistory.shape[0]):
			purchases = self.SalesHistory[user,:]
			items = np.where(purchases==1)[0]
			userf = self.I.ItemsFeatures[items]
			userfm = np.mean(userf,axis=0)
			userfm = userfm/np.max(userfm)
			feat = np.where(userfm>0.33)[0]
			for f in feat: F.append([int(user),int(f)])
		np.savetxt(self.outfolder + "/users_attributes.csv", np.array(F), delimiter=",", fmt='%d')

		# purchases/positive only feedback
		if self.I.activeItemIndeces:
			p = np.where(self.SalesHistory>=1)
			z = zip(p[0],p[1])
			l = [[i,j] for i,j in z if j in self.I.activeItemIndeces]
			np.savetxt(self.outfolder + "/positive_only_feedback.csv", np.array(l), delimiter=",", fmt='%d')

		# export the active items, or all of them if activeItemIndeces is empty
		if not self.I.activeItemIndeces: self.I.activeItemIndeces = [i for i in range(self.I.totalNumberOfItems)]
		d = []
		for i in self.I.activeItemIndeces:
			feat = np.where(self.I.ItemsFeatures[i]/np.max(self.I.ItemsFeatures[i])>0.33)[0]
			for f in feat: d.append([int(i),int(f)])
		np.savetxt(self.outfolder + "/items_attributes.csv", np.array(d), delimiter=",", fmt='%d')
	
	# run MML
	def mmlRecommendation(self):
		# run
		command = "mono MyMediaLite/item_recommendation.exe --training-file=" + self.outfolder + "/positive_only_feedback.csv --item-attributes=" + self.outfolder + "/items_attributes.csv --recommender="+self.algorithm+" --predict-items-number="+str(self.n)+" --prediction-file=" + self.outfolder + "/output.txt --user-attributes=" + self.outfolder + "/users_attributes.csv" # --random-seed="+str(int(self.seed*random.random()))
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

		sns.set_context("notebook", font_scale=1.6, rc={"lines.linewidth": 1.0,'xtick.labelsize': 32, 'axes.labelsize': 32})
		sns.set(style="whitegrid")
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		flatui = sns.color_palette("husl", 8)
		f, ax = matplotlib.pyplot.subplots(1,1, figsize=(6,6), sharey=True)

		# products
		cmaps= ['Blues','Reds','Greens','Oranges','Greys']
		colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(self.I.topics))]
		
		# if no sales history yet, display items with prominence as 3rd dimension
		if np.sum(np.sum(self.SalesHistory))==0:
			n = np.sum(self.SalesHistory,axis=0)
			for i in range(self.I.totalNumberOfItems): 
				color = colors[self.I.ItemsClass[i]]
				s = self.I.ItemProminence[i]*40
				ax.scatter(self.I.Items[i,0], self.I.Items[i,1], marker='o', c=color,s=s,alpha=0.5)
		else:
			# KDE plot
			n = np.sum(self.SalesHistory,axis=0)
			for cat in range(len(self.I.topics)): # 5 topic spaces
				indeces=np.where(self.I.ItemsClass==cat)[0]
				x = []
				for i in indeces:
					if n[i]>0:
						for k in range(int(n[i])): x.append([self.I.Items[i,0],self.I.Items[i,1]])
				ax = sns.kdeplot(np.array(x)[:,0], np.array(x)[:,1], shade=True, shade_lowest=False, alpha = 0.4, cmap=cmaps[cat],kernel='gau')
			
			# scatter
			for i in range(self.I.totalNumberOfItems): 
				color = colors[self.I.ItemsClass[i]]
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
				ax.scatter(self.I.Items[i,0], self.I.Items[i,1], marker=marker, c=color,s=s,alpha=v)	
		
		# final user position as a circle
		for i in range(len(self.U.Users[:,1])):
			ax.scatter(self.U.Users[i,0], self.U.Users[i,1], marker='+', c='k',s=20, alpha = 0.8 )
		
		# user drift
		if drift:
			for i in range(len(self.U.Users[:,1])):
				for j in range(len(self.U.X[i])-1):
					if self.U.X[i][j+1]!=0 and self.U.Y[i][j+1]!=0:
						ax.plot([self.U.X[i][j], self.U.X[i][j+1]], [self.U.Y[i][j], self.U.Y[i][j+1]], 'k-', lw=1, alpha =0.4)

		ax.set_xlabel(self.algorithm)
		ax.set_aspect('equal', adjustable='box')
		ax.set_xlim([-1.1,1.1])
		ax.set_ylim([-1.1,1.1])
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(14) 
		matplotlib.pyplot.tight_layout()
		matplotlib.pyplot.savefig(self.outfolder + "/" + output)
		if not storeOnly: matplotlib.pyplot.show()
 
	def showSettings(self):
		variables = [key for key in self.__dict__.keys() if (type(self.__dict__[key]) is str or type(self.__dict__[key]) is float or type(self.__dict__[key]) is int or type(self.__dict__[key]) is list and len(self.__dict__[key])<10)]
		old = self.__dict__
		Json={ key: old[key] for key in variables }
		print(json.dumps(Json, sort_keys=True, indent=4))


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
	sim.outfolder = outfolder
	sim.totalNumberOfIterations = iterationsPerRecommender*2 # one for the control and one for each rec
	sim.seed = seed
	sim.n = numberOfRecommendations

	printj("Initialize users/items classes...")
	U = users()
	I = items()
	# user/items input arguments
	U.delta, U.totalNumberOfUsers, I.numberOfNewItemsPI = delta, totalNumberOfUsers, newItemsPerIteration
	U.seed = seed
	U.Lambda = focusOnProminentItems
	U.meanSessionSize = meanSessionSize
	I.seed = seed
	I.topicsFrequency = topicweights
	I.topicsSalience = topicprominence
	I.generatePopulation(sim.totalNumberOfIterations)
	U.generatePopulation()
	
	printj("Create simulation instance...")
	sim.U = copy.deepcopy(U)
	sim.I = copy.deepcopy(I)
	sim.createSimulationInstance()
	
	# printj("Plotting users/items in 2d space...")
	# sim.plot2D(storeOnly = True)

	printj("Running Control period...", comments = "We first run a Control period without recommenders to deal with the cold start problem.")
	sim.setAlgorithm("Control")
	sim.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender)])
	sim.showSettings()
	sim.U.showSettings()
	sim.I.showSettings()

	#printj("Saving...", comments = "Output pickle file is stored in your workspace.")
	#pickle.dump(sim.Data, open(sim.outfolder + '/Control-data.pkl', 'wb'))
	printj("Plotting...")
	sim.plot2D(drift = True, output = "2d-Control.pdf")
	
	printj("Running Recommenders....", comments = "The recommenders continue from the Control period.")
	for rec in recommenders:
		sim2 = copy.deepcopy(sim) 	# continue from the control period
		sim2.setAlgorithm(rec)
		sim2.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender,iterationsPerRecommender*2)])
		#printj("Saving for "+rec+"...", comments = "Output pickle file is stored in your workspace.")
		#pickle.dump(sim2.Data, open(sim2.outfolder + '/'+rec+'-data.pkl', 'wb'))
		printj("Plotting for "+rec+"...")
		sim2.plot2D(drift = True, output = "2d-"+sim2.algorithm+".pdf")
      
if __name__ == "__main__":
   main(sys.argv[1:])           



