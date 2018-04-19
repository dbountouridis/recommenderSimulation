import numpy as np
from scipy import spatial
import random
import time
import matplotlib.pyplot as plt


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
		P = np.zeros([A,I]) # Purchases
		H = np.zeros([A,I]) # Histories
	else:
		P = np.zeros([A,I + 1]) # Purchases
		H = np.zeros([A,I + 1]) # Histories

	# Create distance matrix
	D = np.zeros([A,I])
	D = spatial.distance.cdist(IdealPoints, Products)

	# Make binary awareness matrix 
	W = makeawaremx(awaremech,theta,Products,D,A,I,Lambda)

	# ind = np.where(W==1)[1]
	# print(ind)
	# plt.scatter(Products[ind,0], Products[ind,1], marker='o', c='b',s=150,alpha=0.5)
	# plt.show()

	selectedItems = []
	for t in range(iters1):
		for a in range(A):
			indexOfChosenItem = LogitChoiceByHand(D[a,:], metric, k, 0, delta, 1, H[a,:], varBeta, W[a,:])
			selectedItems.append(indexOfChosenItem)
	
	# some plotting
	n, bins, patches = plt.hist(selectedItems, bins=range(I+1),  facecolor='g', alpha=0.75)
	plt.close()
	ax = plt.gca()
	ax.scatter(IdealPoints[:,0], IdealPoints[:,1], marker='^', c='b',s=50)
	ax.scatter(Products[:,0], Products[:,1], marker='o', c='r',s=50)
	for i in range(I):
		plt.scatter(Products[i,0], Products[i,1], marker='o', c='y',s=400*(n[i]/np.max(n)),alpha=0.8)
	circle1 = plt.Circle((0, 0), 1, color='r',fill=False)
	ax.add_artist(circle1)
	plt.show()


## INPUTS
A = 10                         #Agents, users
I = 200                           #Items, products
dim = 2                          #Dimension of map
engine = 1                       #1=CF, 2=Inverse CF, 3=TFIDF-CF, 4=Lowest seller, 5=Median seller, 6=Highest seller, 7=Random, 8=TFIDF + Inv Weight
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
varBeta = 0                     #Variety: coefficient of smooth history in utility function, not mentioned in the paper

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
iters1 = 1000                      #Length of period without recommendations (all agents make 1 purchase/iteration)
iters2 = 1000                      #Length of period with recommendations (uses sales data left at end of Iters1)
plots=0                        #1=produce all plots, 0=plots off

# experiment
IdealPoints = np.array([ [np.random.normal(), np.random.normal()] for i in range(A)])
Products = np.array([ [np.random.normal(), np.random.normal()] for i in range(I)])



sim2fcntimeseries(A,I,iters1,iters2,n,delta,IdealPoints,Products,engine,metric,k,plots,i,spec,varAlpha,varBeta,awaremech,theta,opdist,Lambda);
