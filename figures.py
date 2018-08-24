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
from sklearn.mixture import GaussianMixture
'''
Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r
'''

__author__ = 'Dimitrios Bountouridis'
def pltt(df,output):
	sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.0})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	# plot
	fig, ax = plt.subplots()
	sns.heatmap(df,cmap="binary",ax=ax, cbar=False, xticklabels = 5, yticklabels = 5)
	circle1 = plt.Circle((20, 20), 10, color='k',fill=False,alpha=0.1,zorder=1, edgecolor='k', lw=2)
	ax.text(20.5+10, 10, "Mainstream", ha="center", va="center", size=14, color = "k")
	ax.add_artist(circle1)
	ax.text(user[0]*10+8, abs(user[1])*10+20, "User", ha="center", va="center", size=14, color = "k")
	
	ax.plot([20.5, 20.5], [0,41], 'k--',  lw=2, alpha =0.1)
	ax.plot([0, 41], [20.5,20.5], 'k--',  lw=2, alpha =0.1)
	ax.set_aspect('equal', adjustable='box')
	plt.savefig(output, format='pdf')
	plt.show()

def pltt2(M, r, output):
	sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 1.0})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	x_ = []
	y_ = []
	for i in range(len(r)):
		for j in range(len(r)):
			for k in range(int(M[i,j])):
				x_.append(r[i])
				y_.append(r[j])
	fig, ax = plt.subplots()
	ax= sns.kdeplot(x_, y_,ax=ax,n_levels=30, cmap="Purples_d",xticklabels = 5, yticklabels = 5,joint_kws=dict(shade_lowest=False)) #share=True
	circle1 = plt.Circle((20, 20), 10, color='k',fill=False,alpha=0.6,zorder=1, edgecolor='k', lw=2)
	ax.text(20.5+10, 10, "Mainstream", ha="center", va="center", size=14, color = "k")
	ax.add_artist(circle1)
	ax.text(user[0]*10+8, abs(user[1])*10+20, "User", ha="center", va="center", size=14, color = "k")
	ax.set_aspect('equal', adjustable='box')
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	plt.savefig(output, format='pdf')
	plt.show()

def plotTopics(X,L,X1,L1,classes):
	sns.set_context("notebook", font_scale=1.6, rc={"lines.linewidth": 1.0,'xtick.labelsize': 32, 'axes.labelsize': 32})
	sns.set(style="whitegrid")
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	cmaps= ['Blues','Reds','Greens','Oranges','Greys']
	t = ["entertainment","business","sports","politics","tech"]

	f, ax = plt.subplots()
	ax.set_aspect("equal")
	for i in range(5): # 5 topic spaces
		indeces=np.where(L1==i)[0]
		x = X1[indeces]
		indeces=np.where(L==i)[0]
		x_ = X[indeces]
		ax = sns.kdeplot(x[:,0], x[:,1], shade=True, shade_lowest=False, alpha = 0.7, cmap=cmaps[i],kernel='gau')
		color = sns.color_palette(cmaps[i])[-2]
		plt.scatter(x_[:,0], x_[:,1], c = color, s=5)
	ax.set_aspect('equal', adjustable='box')
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14) 
	ax.set_xlim([-1.1,1.1])
	ax.set_ylim([-1.1,1.1])
	for i in range(5): # 5 topic spaces
		indeces=np.where(L==i)[0]
		x = X[indeces[0]][0]
		y = X[indeces[0]][1]
		if classes[indeces[0]] == "sports":
			x -= 0.25
			y -= 0.66
		if classes[indeces[0]] == "business":
			x = 0.
			y = -0.75
		if classes[indeces[0]] == "politics":
			x = .6
			y = .25
		if classes[indeces[0]] == "entertainment":
			x = -.2
			y = .8
		if classes[indeces[0]] == "tech":
			x = -.1
			y = -.1
		ax.text(x, y, classes[indeces[0]], size=14)
	#sns.set(font_scale = 2)	
	plt.show()

def LogitChoiceByHand(Distances,k):
	return - k*np.log(Distances)

def makeawaremx(theta,Products,Dij,Do,A,I,Lambda=0.75):
	W = np.zeros([A,I])
	for a in range(A):
		for i in range(I):
			W[a,i] = Lambda*np.exp(-(np.power(Do[i],2))/(theta/1)) + (1-Lambda)*np.exp(-(np.power(Dij[a,i],2))/(theta/3))
	return W

# init data
labels = [str(i/10) for i in range(-20,21,1)]
r = [i/10 for i in range(-20,21,1)]
M = np.zeros([len(r), len(r)])
points = []
user = [-1.3, -1.3]
for i,v in enumerate(r):
	for j,v2 in enumerate(r):
		points.append([v,v2])
Do = spatial.distance.cdist([[0,0]], points)[0]
D = spatial.distance.cdist([user], points)

topicSpace = True
Awareness = 0
Vutility = 0 
AwTimesV = 0
 
theta = 0.35    # Awareness Scaling, .35 in paper
Lambda = 0.75   # This is crucial since it controls how much the users focus on mainstream items, 0.75 default value (more focused on mainstream)
k = 10 

if topicSpace:
	random.seed(1)
	(X,labels,classes) = pickle.load(open('BBC data/t-SNE-projection1.pkl','rb'))
	classes = np.array(classes)
	classes[np.where(np.array(classes)=="sport")[0]]="sports"
	gmm = GaussianMixture(n_components=5, random_state =2).fit(X)
	samples_,ItemsClass = gmm.sample(1000)
	Items = samples_/55  # scale down
	ItemFeatures = gmm.predict_proba(samples_)
	plotTopics(np.array(X)/55,np.array(labels),Items,ItemsClass,classes)


if Vutility:
	# compute 
	V = LogitChoiceByHand(D,k)
	
	for index,p in enumerate(points):
		x = r.index(p[0])
		y = r.index(p[1])
		if D[0][index]!=0:
			M[x,y]=- k*np.log(D[0][index])
		else:
			M[x,y]=- k*np.log(0.1)
	M = M -np.min(M)
	M = M/np.sum(M)
	M1 = M.copy()

	M1_ = M1/np.max(M1)
	M1_ = np.round(M1_*50)
	pltt2(M1_, r,"plots/vutility.pdf")


	# to dataframe
	df = pd.DataFrame(M,columns=labels)
	df["Y"] = labels[::-1]
	df = df.set_index('Y')

	# pltt(df,"plots/vutility.pdf")

if Awareness:
	W = makeawaremx(theta,points,D,Do,1,len(points),Lambda)
	for index,p in enumerate(points):
		x = r.index(p[0])
		y = r.index(p[1])
		M[x,y]=W[0][index]
	M = M -np.min(M)
	M = M/np.sum(M)
	M2 = M.copy()

	M2_ = M2/np.max(M2)
	M2_ = np.round(M2_*50)
	pltt2(M2_, r,"plots/awareness.pdf")


	# to dataframe
	df = pd.DataFrame(M,columns=labels)
	df["Y"] = labels[::-1]
	df = df.set_index('Y')
	# pltt(df,"plots/awareness.pdf")

	

if AwTimesV:

	# to dataframe
	M3 = M1*M2
	M3 = M3 -np.min(M3)
	M3 = M3/np.sum(M3)

	M3_ = M3/np.max(M3)
	M3_ = np.round(M3_*50)
	pltt2(M3_, r, "plots/AwTimesV.pdf")

	df = pd.DataFrame(M1*M2,columns=labels)
	df["Y"] = labels[::-1]
	df = df.set_index('Y')
	#pltt(df,"plots/AwTimesV.pdf")




