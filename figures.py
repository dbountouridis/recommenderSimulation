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

Awareness = True
Vutility = True 
AwTimesV = True
 
theta = 0.35    # Awareness Scaling, .35 in paper
Lambda = 0.75   # This is crucial since it controls how much the users focus on mainstream items, 0.75 default value (more focused on mainstream)
k = 10 



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




