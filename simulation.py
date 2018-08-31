""" Simulation of online news consumption including recommendations.

A simulation framework  for the visualization and analysis of the effects of different recommenders systems. 
This simulation draws mainly on the work of Fleder and Hosanagar (2017). To account for the specificities 
of news consumption, it includes both users preferences and editorial priming as they interact in a 
news-webpage context. The simulation models two main components: users (preferences and behavior) 
and items (article content, publishing habits). Users interact with items and are recommended items
based on their interaction.

Example:
	An example 30 sim iterations for 200 users interacting with 100 items per iteration. Five 
	algorithms are of interest in this case Random,WeightedBPRMF,ItemKNN,MostPopular and UserKNN:

	$ python3 simulation.py -d 5 -u 200 -i 30 -t 100 -s 2 
	-r "Random,WeightedBPRMF,ItemKNN,MostPopular,UserKNN" 
	-f "temp" -n 5 -p 0.6 -N 6 -w "0.2,0.2,0.2,0.2,0.2" 
	-g "0.05,0.07,0.03,0.85,0.01"

Todo:
	* Add data export function.

"""

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

__author__ = 'Dimitrios  Bountouridis'

def cdf(weights):
	""" Cummulative density function.

	Used to convert topic weights into probabilities.

	Args:
		weights (list): An array of floats corresponding to weights 

	"""

	total = sum(weights)
	result = []
	cumsum = 0
	for w in weights:
		cumsum += w
		result.append(cumsum / total)
	return result

def selectClassFromDistribution(population, weights):
	""" Given a list of classes and corresponding weights randomly select a class.

	Args:
		population (list): A list of class names e.g. business, politics etc
		weights (list): Corresponding float weights for each class.

	"""

	assert len(population) == len(weights)
	cdf_vals = cdf(weights)
	x = random.random()
	idx = bisect.bisect(cdf_vals, x)
	return population[idx]

def standardize(num, precision = 2):
	""" Convert number to certain precision.

	Args:
		num (float): Number to be converted
		precision (int): Precision either 2 or 4 for now

	"""

	if precision == 2:
		return float("%.2f"%(num))
	if precision == 4:
		return float("%.4f"%(num))

def plotlyjson(x=[],y=[],type_=[],mode="none",title="",ytitle = "",xtitle = ""):
	""" Export data to plotly json schema.

	Args: 
		see plotly https://help.plot.ly/json-chart-schema/

	Returns:
		param1 (dict): Json schema


	"""

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

def printj(text, comments=""):
	json = {"action":text,"comments":comments}
	print(json)

def euclideanDistance(A,B):
	""" Compute the pairwise distance between arrays of (x,y) points.

	We use a numpy version which is C++ based for the sake of efficiency.
	
	"""
	
	#spatial.distance.cdist(A, B, metric = 'euclidean')
	return np.sqrt(np.sum((np.array(A)[None, :] - np.array(B)[:, None])**2, -1)).T


class Users(object):
	""" The class for modeling the user preferences (users) and user behavior.

	The users object can be passed from simulation to simulation, allowing for
	different recommendation algorithms to be applied on. The default attributes
	correspond to findings reports on online news behavior (mostly Mitchell et 
	al 2017,'How Americans encounter, recall and act upon digital news').

	Todo:
		* Allow export of the data for analysis.

	"""

	def __init__(self):
		""" The initialization simply sets the default attributes.

		"""

		self.seed = 1

		self.totalNumberOfUsers = 200  # Total number of users                    
		self.percentageOfActiveUsersPI = 1.0 # Percentage of active users per iterations
 
		self.m = 0.05  # Percentage of the distance_ij covered when a user_i drifts towards an item_j

		# Choice attributes
		self.k = 20                          
		self.delta = 5
		self.beta = 0.9     
		self.meanSessionSize = 6                     

		# Awareness attributes
		self.theta = 0.07  # Proximity decay
		self.thetaDot = 0.5  # Prominence decay
		self.Lambda = 0.6  # Awareness balance between items in proximity and prominent items
		self.w = 40  # Maximum awareness pool size 
		self.Awareness = [] # User-item awareness matrix

		self.Users = []  # User preferences, (x,y) position of users on the attribute space
		self.UsersClass = []  # Users be assigned a class (center at attribute space)
		self.userVarietySeeking = []  # Users' willingness to drift
		self.X = False  # Tracking the X,Y position of users throught the simulation
		self.Y = False

	def generatePopulation(self):
		""" Genererating a population of users (user preferences and variety seeking).

		"""

		random.seed(self.seed)
		np.random.seed(self.seed)
		
		# Position on the attribute space. Uniform, bounded by 1-radius circle
		self.Users = np.random.uniform(-1,1,(self.totalNumberOfUsers,2))
		for i, user in enumerate(self.Users):
			while euclideanDistance([user], [[0,0]])[0][0]>1.1:
				user = np.random.uniform(-1,1,(1,2))[0]
			self.Users[i] = user
	
		# Variety seeking, willingness to drift. Arbitrary defined
		lower, upper = 0, 1
		mu, sigma = 0.1, 0.03
		X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		self.userVarietySeeking = X.rvs(self.totalNumberOfUsers, random_state = self.seed)

		# Users can be assigned a class (most proxiamte attribute center), not currently used.
		#self.UsersClass = [gmm.predict([self.Users[i]*55])[0] for i in range(self.totalNumberOfUsers)]

		self.X = {i:[self.Users[i,0]] for i in range(self.totalNumberOfUsers)}
		self.Y = {i:[self.Users[i,1]] for i in range(self.totalNumberOfUsers)}
 
	def sessionSize(self):
		""" Draw the session size (amount of items to purchase) of each user at each iteration from a normal distribution.

		Returns:
			int: the session size

		"""

		return int(np.random.normal(self.meanSessionSize, 2))

	def subsetOfAvailableUsers(self):
		""" Randomly select a subset of the users.

		"""

		self.activeUserIndeces = np.arange(self.totalNumberOfUsers).tolist()
		random.shuffle(self.activeUserIndeces)
		self.activeUserIndeces = self.activeUserIndeces[:int(len(self.activeUserIndeces)*self.percentageOfActiveUsersPI)] 
		self.nonActiveUserIndeces = [ i  for i in np.arange(self.totalNumberOfUsers) if i not in self.activeUserIndeces]

	def computeAwarenessMatrix(self, Dij, ItemProminence, activeItemIndeces):
		""" Compute awareness from proximity and prominence (not considering availability, recommendations, history).

		Args:
			Dij (nparray): |Users| x |Items| distance matrix
			ItemProminence (nparray): |Items|-sized prominence vector 

		"""

		totalNumberOfItems = ItemProminence.shape[0]

		W = np.zeros([self.totalNumberOfUsers,totalNumberOfItems])
		W2 = W.copy() # for analysis purposes
		W3 = W.copy() # for analysis purposes
		for a in self.activeUserIndeces:
			W[a,activeItemIndeces] = self.Lambda*(-self.thetaDot*np.log(1-ItemProminence[activeItemIndeces])) + (1-self.Lambda)*np.exp(-(np.power(Dij[a,activeItemIndeces],2))/self.theta)
			W2[a,activeItemIndeces] = self.Lambda*(-self.thetaDot*np.log(1-ItemProminence[activeItemIndeces])) 
			W3[a,activeItemIndeces] = (1-self.Lambda)*np.exp(-(np.power(Dij[a,activeItemIndeces],2))/self.theta)
		R = np.random.rand(W.shape[0],W.shape[1])
		W = R<W
		self.Awareness, self.AwarenessOnlyPopular, self.AwarenessProximity =  W, W2, W3

	def choiceModule(self, Rec, w, distanceToItems, sessionSize, control = False):
		""" Selecting items to purchase for a single user.

		Args:
			Rec (list): List of items recommended to the user
			w (nparray): 1 x |Items| awareness of the user
			distanceToItems (nparray): 1 x |Items| distance of the user to the items
			sessionSize (int): number of items that the user will purchase

		Returns:
			 param1 (list): List of items that were selected including the stochastic component
			 param2 (list): List of items that were selected not including the stochastic component

		"""

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
	
	def computeNewPositionOfUser(self, user, ChosenItems):
		""" Compute new position of a user given their purchased item(s).

		Args:
			user (int): Index of specific user.
			ChosenItems (list): (x,y) position array of items selected by the user.

		"""

		for itemPosition in ChosenItems:
			dist =  euclideanDistance([self.Users[user]], [itemPosition])[0]
			p = np.exp(-(np.power(dist,2))/(self.userVarietySeeking[user])) # based on the awareness formula
			B = np.array(self.Users[user])
			P = np.array(itemPosition)
			BP = P - B
			x,y = B + self.m*(random.random()<p)*BP 
			self.Users[user] = [x,y]
		self.X[user].append(x)
		self.Y[user].append(y)

	def showSettings(self):
		""" A simple function to print most of the attributes of the class.

		"""

		variables = [(key,type(self.__dict__[key])) for key in self.__dict__.keys() if (type(self.__dict__[key]) is str or type(self.__dict__[key]) is float or type(self.__dict__[key]) is int )]

		Json={ key: self.__dict__[key] for key,tp in variables }
		print(json.dumps(Json, sort_keys=True, indent=4))

class Items(object):
	""" The class for modeling the items' content (items) and prominence.

	The items object can be passed from simulation to simulation, allowing for
	different recommendation algorithms to be applied on. The default attributes
	correspond to findings reports on online news behavior (mostly Mitchell et 
	al 2017,'How Americans encounter, recall and act upon digital news').

	Todo:
		* Allow export of the data for analysis.

	"""
	def __init__(self):
		""" The initialization simply sets the default attributes.

		"""
		self.seed = 1
		self.numberOfNewItemsPI = 100  # The number of new items added per iteration
		self.totalNumberOfItems = False  # The total number of items (relates to the number of iterations)
		self.percentageOfActiveItems = False  
                    		
		# Topics, frequency weights and prominence weights. We use topics instead of "classes" here.
		self.topics = ["entertainment","business","sport","politics","tech"]
		self.topicsProminence = [0.05,0.07,0.03,0.85,0.01] 
		self.topicsFrequency = [0.2, 0.2, 0.2, 0.2, 0.2]

		self.p = 0.1  # Slope of salience decrease function

		self.Items = []  # The items' content (x,y) position on the attribute space
		self.ItemsClass = []  # The items' class corresponds to the most prominent topic
		self.ItemsFeatures = False  # The items' feature vector
		self.ItemsDistances = False  # |Items|x|Items| distance matrix
		self.ItemsOrderOfAppearance = False  # Random order of appearance at each iteration
		self.ItemProminence = False  #  Item's prominence
		self.ItemLifespan = False  # Items' age (in iterations)
		self.hasBeenRecommended = False  # Binary matrix holding whether each items has been recommended

	def generatePopulation(self, totalNumberOfIterations):
		""" Genererating a population of items (items' content and initial prominence).

		"""

		random.seed(self.seed)
		np.random.seed(self.seed)

		# Compute number of total items in the simulation
		self.totalNumberOfItems = totalNumberOfIterations*self.numberOfNewItemsPI                    
		self.percentageOfActiveItems = self.numberOfNewItemsPI/self.totalNumberOfItems

		# Apply GMM on items/articles from the BBC data
		R, S = [5,1,6,7], [5,2,28,28]
		r = int(random.random()*4)
		printj("Item space projection selected:",R[r])
		(X,labels,topicClasses) = pickle.load(open('BBC data/t-SNE-projection'+str(R[r])+'.pkl','rb'))
		gmm = GaussianMixture(n_components=5, random_state=S[r]).fit(X)
		
		# Normalize topic weights to sum into 1 (CBF)
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
		self.ItemsClass = np.array(self.ItemsClass)
		self.ItemsFeatures = gmm.predict_proba(self.Items)
		self.Items = self.Items/55  # Scale down to -1, 1 range
		
		# Cosine distance between item features
		self.ItemsDistances = spatial.distance.cdist(self.ItemsFeatures, self.ItemsFeatures, metric='cosine')

		# Generate a random order of item availability
		self.ItemsOrderOfAppearance = np.arange(self.totalNumberOfItems).tolist()
		random.shuffle(self.ItemsOrderOfAppearance)
	
		# Initial prominence
		self.initialProminceZ0()
		self.ItemProminence = self.ItemsInitialProminence.copy()

		# Lifespan, item age
		self.ItemLifespan = np.ones(self.totalNumberOfItems)

		# Has been recommended before
		self.hasBeenRecommended = np.zeros(self.totalNumberOfItems)

	def prominenceFunction(self, initialProminence, life):
		""" Decrease of item prominence, linear function.

		Args:
			initialProminence (float): The initial prominence of the item
			life (int): The item's age (in iterations)

		Returns:
			param1 (float): New prominence value

		"""

		x = life
		y = (-self.p*(x-1)+1)*initialProminence
		return max([y, 0])

	def subsetOfAvailableItems(self,iteration):
		""" Randomly select a subset of the items. 

		The random order of appearance has already been defined in ItemsOrderOfAppearance. The function simply 
		extends the size of the activeItemIndeces array.

		Args:
			iteration (int): the current simulation iteration

		"""

		self.activeItemIndeces =[j for j in self.ItemsOrderOfAppearance[:(iteration+1)*int(self.totalNumberOfItems*self.percentageOfActiveItems)] if self.ItemProminence[j]>0]
		self.nonActiveItemIndeces = [ i  for i in np.arange(self.totalNumberOfItems) if i not in self.activeItemIndeces]

	def updateLifespanAndProminence(self):
		""" Update the lifespan and promince of the items.

		"""

		self.ItemLifespan[self.activeItemIndeces] = self.ItemLifespan[self.activeItemIndeces]+1
		
		for a in self.activeItemIndeces:
			self.ItemProminence[a] = self.prominenceFunction(self.ItemsInitialProminence[a],self.ItemLifespan[a])
		
	def initialProminceZ0(self):
		""" Generate initial item prominence based on the topic weights and topic prominence.

		"""

		self.topicsProminence = [np.round(i,decimals=2) for i in self.topicsProminence/np.sum(self.topicsProminence)]
		counts = dict(zip(self.topics, [len(np.where(self.ItemsClass==i)[0]) for i,c in enumerate(self.topics) ]))
		items = len(self.ItemsClass)
		population = self.topics
		
		# Chi square distribution with two degrees of freedom. Other power-law distributions can be used.
		df = 2
		mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
		x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), items)
		rv = chi2(df)
		
		Z = {}
		for c in self.topics: Z.update({c:[]})		
		
		# Assign topic to z prominence without replacement
		for i in rv.pdf(x):
			c = selectClassFromDistribution(population, self.topicsProminence)
			while counts[c]<=0:
				c = selectClassFromDistribution(population, self.topicsProminence)
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
		""" A simple function to print most of the attributes of the class.

		"""
		variables = [key for key in self.__dict__.keys() if (type(self.__dict__[key]) is str or type(self.__dict__[key]) is float or type(self.__dict__[key]) is int or type(self.__dict__[key]) is list and len(self.__dict__[key])<10)]
		old = self.__dict__
		Json={ key: old[key] for key in variables }
		print(json.dumps(Json, sort_keys=True, indent=4))

class Simulation(object):
	""" The simulation class takes users and items and simulates their interaction.

	The simulation can include recommendations (currently using a MyMediaLite wrapper).
	Alternative toolboxes can be used. The simulation class also stores results for 
	analysis and computes diversity metrics (based on the PhD thesis of Vargas).

	"""

	def __init__(self):	
		self.totalNumberOfIterations = 20
		self.n = 5  # Number of recommended items per iteration per user
		self.algorithm = "Control"	# The current recommendation algorithm
		self.AnaylysisInteractionData = []  # Holder for results/data

		self.D = []  # Distance matrix |Users|x|Items| between items and users
		self.SalesHistory = []  # User-item interaction matrix |Users|x|Items|7......7

		self.diversityMetrics = {}  # Holder for diversity metrics (means + std)
		for key in ["EPC", "EPCstd",'ILD',"Gini", "EFD", "EPD", "EILD", 'ILDstd', "EFDstd", "EPDstd", "EILDstd"]:
			self.diversityMetrics.update({key:[]})

		self.outfolder = ""


	def createSimulationInstance(self):
		""" Create an instance of the simulation by computing users to items distances.

		Todo:
			* Updating the item-distance only for items that matter

		"""

		
		self.D =  euclideanDistance(self.U.Users, self.I.Items)
		self.SalesHistory = np.zeros([self.U.totalNumberOfUsers,self.I.totalNumberOfItems]) 						
	
	def exportAnalysisDataAfterIteration(self):
		""" Export some analysis data to dataframes.

		"""


		printj("Exporting per iteration data...", comments = 'Two output pickle files are stored in your workspace.')
		df = pd.DataFrame(self.AnaylysisInteractionData,columns=["Iteration index",
			"User",
			"MML method",
			"Item",
			"Item Age",
			"Item Prominence",
			"Class/Topic",
			"Was Recommended",
			"Agreement between deterministic and stochastic choice", 
			"Item has been recommended before",
			"Class/Topic agreement between deterministic and stochastic choice", 
			"Class/Topic agreement between choice and users main topic",
			"User class",
			"InInitialAwareness"])
		df.to_pickle(self.outfolder + "/dataframe for simple analysis-"+self.algorithm+".pkl")

		# Metrics output
		df = pd.DataFrame(self.diversityMetrics)
		df["Iteration index"] = np.array([i for i in range(len(self.diversityMetrics["EPC"])) ])
		df["MML method"] = np.array([self.algorithm for i in  range(len(self.diversityMetrics["EPC"]))])
		df.to_pickle(self.outfolder + "/metrics analysis-"+self.algorithm+".pkl")

	def exportJsonForOnlineInterface(self, epoch, epoch_index, iterationRange, SalesHistoryBefore):
		""" Export some data in json for the SIREN online interface and for terminal inspection.

		Args:
			epoch (int): The current simulatio iteration
			epoch_index (int): The index of the simulation iteration
			iterationRange (list)
			SalesHistoryBefore: The purchase history before the current iteration


		"""

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

		# Output on terminal
		print(json.dumps(Json, sort_keys=True, indent=4))


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
		
		# Diversity figures
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
		
		# Output in file
		Json.update({"Users position" : [(standardize(i[0]),standardize(i[1])) for i in self.U.Users]})
		Json.update({"Items position" : [(standardize(i[0]),standardize(i[1])) for i in self.I.Items]})
		json.dump(Json, open(self.outfolder + '/'+str(self.algorithm)+'-data.json', 'w'),sort_keys=True, indent=4)
	
	def awarenessModule(self, epoch):
		""" This function computes the awareness of each user.

		While the proximity/prominence awareness is computed in the user class, the current function
		updates that awareness to accommodate for the non-available items and those that the user
		has purchased before. The number of items in the awareness is also limited.

		Args:
			epoch (int): The current iteration. 
		
		"""

		self.U.subsetOfAvailableUsers()
		self.I.subsetOfAvailableItems(epoch)
		self.U.computeAwarenessMatrix(self.D, self.I.ItemProminence, self.I.activeItemIndeces)
		
		# Adjust for availability 
		self.U.Awareness[:,self.I.nonActiveItemIndeces] = 0 

		# Adjust for purchase history
		self.U.Awareness = self.U.Awareness - self.SalesHistory>0

		# Adjust for maximum number of items in awareness
		for a in range(self.U.totalNumberOfUsers):
			w = np.where(self.U.Awareness[a,:]==1)[0]
			if len(w)>self.U.w:
				windex = w.tolist()
				random.shuffle(windex)
				self.U.Awareness[a,:] = np.zeros(self.I.totalNumberOfItems)
				self.U.Awareness[a,windex[:self.U.w]] = 1	
		
	def temporalAdaptationsModule(self, epoch):
		""" Update the user-items distances and item- lifespand and prominence.

		Todo:
			* Updating the item-distance only for items that matter

		"""	
	
		self.I.updateLifespanAndProminence()

		# We compute this here so that we update the distances between users and not all the items
		self.I.subsetOfAvailableItems(epoch+1)
	

		if self.algorithm is not "Control":
			D =  euclideanDistance(self.U.Users, self.I.Items[self.I.activeItemIndeces])
			# If you use only a percentage of users then adjust this function
			for u in range(self.U.totalNumberOfUsers): self.D[u,self.I.activeItemIndeces] = D[u,:]
		

	def runSimulation(self, iterationRange =[]):
		""" The main simulation function.

		For different simulation instantiations to run on the same random order of items
		the iterationRange should be the same.

		Args:
			iterationRange (list): The iteration range for the current simulation

		"""
			
		for epoch_index, epoch in enumerate(iterationRange):

			SalesHistoryBefore = self.SalesHistory.copy()

			printj(self.algorithm+": Awareness...")				
			self.awarenessModule(epoch)
			InitialAwareness = self.U.Awareness.copy()

			# Recommendation module 
			if self.algorithm is not "Control":
				printj(self.algorithm+": Recommendations...")

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
						self.U.Awareness[user, Rec] = 1

						# If recommended but previously purchased, minimize the awareness
						self.U.Awareness[user, np.where(self.SalesHistory[user,Rec]>0)[0] ] = 0  	

			# Choice 
			printj(self.algorithm+": Choice...")
			for user in self.U.activeUserIndeces:
				Rec=np.array([-1])
				
				if self.algorithm is not "Control":
					if user not in recommendations.keys():
						printj(" -- Nothing to recommend -- to user ",user)
						continue
					Rec = recommendations[user]
				
				indecesOfChosenItems,indecesOfChosenItemsW =  self.U.choiceModule(Rec, 
					self.U.Awareness[user,:], 
					self.D[user,:], 
					self.U.sessionSize(), 
					control = self.algorithm=="Control")

				# Add item purchase to histories
				self.SalesHistory[user, indecesOfChosenItems] += 1		
						
				# Compute new user position 
				if self.algorithm is not "Control" and len(indecesOfChosenItems)>0:
					self.U.computeNewPositionOfUser(user, self.I.Items[indecesOfChosenItems])

				# Store some data for analysis
				for i,indexOfChosenItem in enumerate(indecesOfChosenItems):
					indexOfChosenItemW = indecesOfChosenItemsW[i]
					self.AnaylysisInteractionData.append([epoch_index, 
						user, 
						self.algorithm,
						indexOfChosenItem,
						self.I.ItemLifespan[indexOfChosenItem], 
						self.I.ItemProminence[indexOfChosenItem],
						self.I.topics[self.I.ItemsClass[indexOfChosenItem]],
						indexOfChosenItem in Rec, 
						indexOfChosenItem == indexOfChosenItemW,
						self.I.hasBeenRecommended[indexOfChosenItemW],
						self.I.ItemsClass[indexOfChosenItem]==self.I.ItemsClass[indexOfChosenItemW] , 
						0,
						0, 
						InitialAwareness[user,indexOfChosenItem] ])

			# Temporal adaptations
			printj(self.algorithm+": Temporal adaptations...")	
			self.temporalAdaptationsModule(epoch)

			# Compute diversity metrics		
			if self.algorithm is not "Control":
				printj(self.algorithm+": Diversity metrics...")
				
				met = metrics.metrics(SalesHistoryBefore, recommendations, self.I.ItemsFeatures, self.I.ItemsDistances, self.SalesHistory)
				met.update({"Gini": metrics.computeGinis(self.SalesHistory,self.ControlHistory)})
				for key in met.keys():
					self.diversityMetrics[key].append(met[key])

			# Show stats on screen and save json for interface
			self.exportJsonForOnlineInterface(epoch, epoch_index, iterationRange, SalesHistoryBefore)

		# Save results
		self.exportAnalysisDataAfterIteration()
		
	def exportToMMLdocuments(self):
		""" Export users' features, items' content and user-item purchase history for MyMediaLite.

		MyMediaLite has a specific binary input format for user-, item-attributes: the attribute
		either belongs or does not belong to an item or user. To accommodate for that we had to 
		take some liberties and convert the user's feature vector and item's feature vector into
		a binary format.


		"""

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

		if self.I.activeItemIndeces:
			p = np.where(self.SalesHistory>=1)
			z = zip(p[0],p[1])
			l = [[i,j] for i,j in z if j in self.I.activeItemIndeces]
			np.savetxt(self.outfolder + "/positive_only_feedback.csv", np.array(l), delimiter=",", fmt='%d')

		if not self.I.activeItemIndeces: self.I.activeItemIndeces = [i for i in range(self.I.totalNumberOfItems)]
		d = []
		for i in self.I.activeItemIndeces:
			feat = np.where(self.I.ItemsFeatures[i]/np.max(self.I.ItemsFeatures[i])>0.33)[0]
			for f in feat: d.append([int(i),int(f)])
		np.savetxt(self.outfolder + "/items_attributes.csv", np.array(d), delimiter=",", fmt='%d')
	
	def mmlRecommendation(self):
		""" A wrapper around the MyMediaLite toolbox

		Returns:
			recommendations (dict): A {user:[recommended items]} dictionary 
		
		"""

		command = "mono MyMediaLite/item_recommendation.exe --training-file=" + self.outfolder + "/positive_only_feedback.csv --item-attributes=" + self.outfolder + "/items_attributes.csv --recommender="+self.algorithm+" --predict-items-number="+str(self.n)+" --prediction-file=" + self.outfolder + "/output.txt --user-attributes=" + self.outfolder + "/users_attributes.csv" # --random-seed="+str(int(self.seed*random.random()))
		os.system(command)
		
		# Parse output
		f = open( self.outfolder + "/output.txt","r").read() 
		f = f.split("\n")
		recommendations = {}
		for line in f[:-1]:
			l = line.split("\t")
			user_id = int(l[0])
			l1 = l[1].replace("[","").replace("]","").split(",")
			rec = [int(i.split(":")[0]) for i in l1]
			recommendations.update({user_id:rec})
		return recommendations 
		    
	def plot2D(self, drift = False, output = "initial-users-products.pdf", storeOnly = True):
		""" Plotting the users-items on the attribute space.

		Args:
			drift (bool): Whether the user drift should be plotted (it is time consuming)
			output (str): The output pdf file
			storeOnly (bool): Whether the plot should be shown

		"""

		sns.set_context("notebook", font_scale=1.6, rc={"lines.linewidth": 1.0,'xtick.labelsize': 32, 'axes.labelsize': 32})
		sns.set(style="whitegrid")
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		flatui = sns.color_palette("husl", 8)
		f, ax = matplotlib.pyplot.subplots(1,1, figsize=(6,6), sharey=True)

		cmaps= ['Blues','Reds','Greens','Oranges','Greys']
		colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(self.I.topics))]
		
		# If no sales history yet, display items with prominence as 3rd dimension
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
			
			# Scatter
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
		
		# Final user position as a circle
		for i in range(len(self.U.Users[:,1])):
			ax.scatter(self.U.Users[i,0], self.U.Users[i,1], marker='+', c='k',s=20, alpha = 0.8 )
		
		# User drift
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
		""" A simple function to print most of the attributes of the class.

		"""

		variables = [key for key in self.__dict__.keys() if (type(self.__dict__[key]) is str or type(self.__dict__[key]) is float or type(self.__dict__[key]) is int or type(self.__dict__[key]) is list and len(self.__dict__[key])<10)]
		old = self.__dict__
		Json={ key: old[key] for key in variables }
		print(json.dumps(Json, sort_keys=True, indent=4))

def main(argv):
	helpText = 'simulation.py  -i <iterationsPerRecommender> -s <seed> -u <totalusers> -d <deltasalience> -r <recommenders> -t <newItemsPerIteration> -f <outfolder> -n <numberOfRecommendations> -p <focusonprominence> -N <meanSessionSize> -w <topicweights> -g <topicprominence>'
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
	sim = Simulation()
	sim.outfolder = outfolder
	sim.seed = seed
	sim.n = numberOfRecommendations
	
	# The totalNumberOfIterations controls the amount of
	# items that will be generated. We first need to run a Control period for
	# iterarionsPerRecommender iterations, on different items than during the 
	# recommendation period, as such the total amount of iterations is doubled.
	sim.totalNumberOfIterations = iterationsPerRecommender*2 
	
	printj("Initialize users/items classes...")
	U = Users()
	I = Items()

	U.delta = delta
	U.totalNumberOfUsers = totalNumberOfUsers
	U.seed = seed
	U.Lambda = focusOnProminentItems
	U.meanSessionSize = meanSessionSize
	
	I.seed = seed
	I.topicsFrequency = topicweights
	I.topicsSalience = topicprominence
	I.numberOfNewItemsPI = newItemsPerIteration
	
	I.generatePopulation(sim.totalNumberOfIterations)
	U.generatePopulation()
	
	printj("Create simulation instance...")
	sim.U = copy.deepcopy(U)
	sim.I = copy.deepcopy(I)
	sim.createSimulationInstance()
	
	# printj("Plotting users/items in 2d space...")
	# sim.plot2D(storeOnly = True)

	printj("Running Control period...", comments = "We first run a Control period without recommenders to deal with the cold start problem.")
	sim.algorithm = "Control"
	sim.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender)])
	sim.showSettings()
	sim.U.showSettings()
	sim.I.showSettings()

	#printj("Saving...", comments = "Output pickle file is stored in your workspace.")
	#pickle.dump(sim.Data, open(sim.outfolder + '/Control-data.pkl', 'wb'))
	#printj("Plotting...")
	#sim.plot2D(drift = True, output = "2d-Control.pdf")
	
	printj("Running Recommenders....", comments = "The recommenders continue from the Control period.")
	for rec in recommenders:

		# Set the same settings as the control period
		simR = Simulation()
		simR.outfolder = outfolder
		simR.seed = seed
		simR.n = numberOfRecommendations
		simR.totalNumberOfIterations = iterationsPerRecommender*2
		simR.algorithm =  rec
		
		# Copy the users, items and their interactions from the control period
		simR.U = copy.deepcopy(sim.U)
		simR.I = copy.deepcopy(sim.I)
		simR.D = sim.D.copy()  # Start from the control distances between items and users
		simR.SalesHistory = sim.SalesHistory.copy()  # Start from the control sale history
		simR.ControlHistory = sim.SalesHistory.copy()  # We use a copy of the control sales history for the Gini coeff metric
		simR.runSimulation(iterationRange = [i for i in range(iterationsPerRecommender,iterationsPerRecommender*2)])
		
		#printj("Saving for "+rec+"...", comments = "Output pickle file is stored in your workspace.")
		#pickle.dump(sim2.Data, open(sim2.outfolder + '/'+rec+'-data.pkl', 'wb'))
		#printj("Plotting for "+rec+"...")
		#sim2.plot2D(drift = True, output = "2d-"+sim2.algorithm+".pdf")
      
if __name__ == "__main__":
   main(sys.argv[1:])           



