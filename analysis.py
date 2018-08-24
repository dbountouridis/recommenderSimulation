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
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import glob
import sys, getopt
from skbio import DistanceMatrix
from skbio.tree import nj

__author__ = 'Dimitrios Bountouridis'

# diversity figures as they appear on the paper
def diversityAnalysis(infolder):
	print("Simple diversity analysis...")
	# Read files
	files = glob.glob(infolder + "/metrics analysis-*.pkl")
	for i,file in enumerate(files):
		print(i,file)
		if i==0:
			df = pd.read_pickle(file)
		else:
			df_ = pd.read_pickle(file)
			df = df.append(df_, ignore_index=True)
	print("Dataframe:")
	print(df.describe())
	df2 = df.loc[df['MML method'] != "Control"]

	for metric in ["EPC","EPD","EFD","ILD","EILD"]:
		methods = [ g for g, group in df2.groupby("MML method") if g not in ["Random","MostPopular"]]

		if metric == "EPC":
			grouped = ["SoftMarginRankingMF","UserAttributeKNN","MostPopularByAttributes","BPRMF","ItemAttributeKNN","LeastSquareSLIM","MultiCoreBPRMF"]

		L = []
		for i,method1 in enumerate(grouped):
			df1_ = df2.loc[df2['MML method'] == method1]
			data1 = np.array(df1_[metric])
			L.append(data1)
	
		yp = np.max(np.array(L),axis=0).tolist() + np.min(np.array(L),axis=0).tolist()[::-1]
		xp = np.arange(np.array(L).shape[1]).tolist() + np.arange(np.array(L).shape[1]).tolist()[::-1]
		xp = np.array(xp)
		# print(y,x)
		# time.sleep(100)
		# set sns context
		sns.set_context("notebook", font_scale=1.35, rc={"lines.linewidth": 1.2})
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		flatui = sns.color_palette("Dark2", 6)[::-1]
		sns.set_palette(flatui)
		[methods.remove(i) for i in grouped] 
		methods.append("Random")
		methods.append("MostPopular")
		fig, ax = plt.subplots(figsize=(9,6))
		ls = '-'
		k = 0
		for g, group in df2.groupby("MML method"):
			if g not in methods: continue
			x = np.array(group["Iteration index"])
			y = np.array(group[metric])
			yerror = np.array(group[metric+"std"])/2
			ax.errorbar(x+0.02*k, y, yerr=yerror, linestyle=ls, marker='o', markersize=8, label=g)
			k+=1
		ax.set_xticks(np.arange(len(x)) )
		ax.set_xlabel('Iterations')
		ax.set_ylabel(metric)
		ax.set_xlim(xmin=1)
		ax.fill(xp, yp, "g",alpha=0.3, label = "Group A", joinstyle="round")
		ax.legend()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.show()

	# plot
	df2 = df.loc[df['MML method'] != "Control"]
	types = ["LongTail","LongTail","Unexpect","Unexpect","Unexpect","Business"]
	for i,metric in enumerate(["EPC","EFD","EPD","EILD", "ILD","Gini"]):
		g = sns.factorplot(x="Iteration index", y=metric, hue="MML method", data=df2, capsize=.2, palette=flatui, size=6, aspect=1, sharey = False, legend = True, dodge=True, legend_out=False)
		g.despine(left=True)
		plt.savefig(infolder + "/Analysis diversity "+types[i]+" - "+metric+".pdf")


def analysis(infolder):
	print("Simple analysis...")
	# Read files
	files = glob.glob(infolder + "/dataframe for simple analysis-*.pkl")
	for i,file in enumerate(files):
		print(i,file)
		if i==0:
			df = pd.read_pickle(file)
		else:
			df_ = pd.read_pickle(file)
			df = df.append(df_, ignore_index=True)
	print("Dataframe:")
	print(df.describe())
	#print(df)

	# Sanity check 1
	print("Control period: item age distribution")
	D = {}
	for i in range(1,11): D.update({i:[]})

	for g, group in df.groupby("MML method"):
		if g!="Control": continue
		total = np.array(group).shape[0]
		for i,group3 in group.groupby("Item Age"):
			total2 = np.array(group3).shape[0]
			D[i].append(total2/total)
			#print(i,total2/total)
	for key in D.keys():
		print(key,np.mean(D[key]))

	# Sanity check 2
	print("Control period: topic distribution")
	for g, group in df.groupby("MML method"):
		if g!="Control": continue
		total = np.array(group).shape[0]
		for i,group3 in group.groupby("Class/Topic"):
			total2 = np.array(group3).shape[0]
			print(i,total2/total)
	time.sleep(10)
		


	# set sns context
	sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	flatui = sns.color_palette("husl", 8)
	
	for g, group in df.groupby("MML method"):
		print(g)
		D = []
		for i,group3 in group.groupby("Class/Topic"):
			print(i)
			total = np.array(group3).shape[0]
			for t,group2 in group3.groupby("User class"):
				if t==i:
					print(np.array(group2).shape[0]/total)
					D.append(np.array(group2).shape[0]/total)
		print(np.mean(D),np.std(D))
	
	# # plot
	# cmaps= ['Blues','Reds','Greens','Oranges','Greys']
	# t = ["entertainment","business","sport","politics","tech"]
	# colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(t))]
	# g = sns.factorplot(x="Iteration index", y="Counts", hue="Class/Topic", col="MML method", data=df_, capsize=.2, palette=colors, size=3, aspect=2, sharey = True, legend = True, dodge=True, legend_out=False, hue_order = t)
	# g.despine(left=True)
	# plt.savefig(infolder + "/Analysis - Selected articles that were recommended.pdf")
	# #plt.show()
	df_ = df.loc[df['MML method'] != "Control"]
	df2 = df_.loc[df_['InInitialAwareness'] == False]
	# print(np.array(df2["Item"]).shape[0])
	# df2 = df_.loc[df_['InInitialAwareness'] == True]
	# print(np.array(df2["Item"]).shape[0])
	W = []
	for g_, group_ in df.groupby("MML method"):
		D = []
		D_= []
		if g_=="Control": continue
		for g, group in group_.groupby("Item"):
			total = np.array(group).shape[0]
			wr = np.where(group["Was Recommended"]==1)[0].shape[0]
			wnr = np.where(group["Was Recommended"]==0)[0].shape[0]

			D.append(wr/total)
			W.append(wr/total)
			D_.append(wnr/total)
			# time.sleep(1)
			# for i,group3 in group.groupby("Was Recommended"):
			# 	if not i: 
			# 		D.append(np.array(group3).shape[0]/total)
		print(g_,":",np.mean(D),np.std(D),D[:10])
		print(g_,":",np.mean(D_),np.std(D_),D_[:10])
	print("Overall rec:",np.mean(W),np.std(W))
	


	g = sns.factorplot("Was Recommended",col="MML method", kind="count", data=df, capsize=.2, palette=sns.color_palette("BuGn_r",15), size=3, aspect=.75,  legend = True, legend_out=False,dodge=True)
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis - Was Recommended.pdf")

	g = sns.factorplot(y="Agreement between deterministic and stochastic choice",x="MML method", kind="bar", data=df, capsize=.2, palette=flatui, size=4, aspect=1.5,  legend = True, legend_out=False,dodge=True, orient = "v")
	plt.savefig(infolder + "/Analysis - Agreement between deterministic and stochastic choice.pdf")

	g = sns.factorplot(y="Class/Topic agreement between deterministic and stochastic choice",x="MML method", kind="bar", data=df, capsize=.2, palette=flatui, size=4, aspect=1.5,  legend = True, legend_out=False,dodge=True, orient = "v")
	plt.savefig(infolder + "/Analysis - Class-Topic Agreement between deterministic and stochastic choice.pdf")
	#plt.show()

	
	g = sns.factorplot(y="Class/Topic agreement between choice and users main topic",x="MML method", kind="bar", data=df, capsize=.2, palette=flatui, size=4, aspect=1.5,  legend = True, legend_out=False,dodge=True, orient = "v")
	plt.savefig(infolder + "/Analysis - Class-Topic agreement between choice and users main topic.pdf")
	#plt.show()

	g = sns.factorplot("Item Age",col="MML method", kind="count", data=df, capsize=.2, palette=sns.color_palette("BuGn_r",15), size=3, aspect=.75,  legend = True, legend_out=False,dodge=True)
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis - Item Age.pdf")
	#plt.show()

	cmaps= ['Blues','Reds','Greens','Oranges','Greys']
	t = ["entertainment","business","sport","politics","tech"]
	colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(t))]
	df["Item Prominence Quantized"] = np.digitize( df["Item Prominence"], bins = [0.1*i for i in range(0,10)]  )
	g = sns.factorplot("Item Prominence Quantized",col="MML method", kind="count", hue="Class/Topic", data=df, capsize=.2, palette=colors, size=3, aspect=.75,  legend = True, legend_out=False,dodge=True, sharex=True, hue_order=t)
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis - Item Prominence Quantized.pdf")
	#plt.show()

	g = sns.factorplot(x="MML method", hue="Class/Topic",kind="count", data=df, capsize=.2, palette=colors, size=4, aspect=2,  legend = True, dodge=True, legend_out=False, hue_order=t)
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis - Class-Topic.pdf")
	print("Plots stored in " + infolder +".")


# diversity figures as they appear on the paper
def diversityAnalysis2(infolder,infolder2):
	print("Simple diversity analysis...")
	# Read files
	files = glob.glob(infolder + "/metrics analysis-*.pkl")
	for i,file in enumerate(files):
		print(i,file)
		if i==0:
			df = pd.read_pickle(file)
		else:
			df_ = pd.read_pickle(file)
			df = df.append(df_, ignore_index=True)
	print("Dataframe:")
	print(df.describe())
	df2 = df.loc[df['MML method'] != "Control"]

	files = glob.glob(infolder2 + "/metrics analysis-*.pkl")
	for i,file in enumerate(files):
		print(i,file)
		if i==0:
			df = pd.read_pickle(file)
		else:
			df_ = pd.read_pickle(file)
			df = df.append(df_, ignore_index=True)
	print("Dataframe:")
	print(df.describe())
	df2b = df.loc[df['MML method'] != "Control"]

	for metric in ["EPC","EPD","EFD","ILD","EILD"]:
		methods = [ g for g, group in df2.groupby("MML method") if g not in ["Random","MostPopular"]]

		if metric == "EPC":
			grouped = ["BPRSLIM","SoftMarginRankingMF","UserAttributeKNN","MostPopularByAttributes","BPRMF","ItemAttributeKNN","LeastSquareSLIM","MultiCoreBPRMF"]

		L = []
		for i,method1 in enumerate(grouped):
			df1_ = df2.loc[df2['MML method'] == method1]
			data1 = np.array(df1_[metric])
			L.append(data1)
	
		yp = np.max(np.array(L),axis=0).tolist() + np.min(np.array(L),axis=0).tolist()[::-1]
		xp = np.arange(np.array(L).shape[1]).tolist() + np.arange(np.array(L).shape[1]).tolist()[::-1]
		xp = np.array(xp)
		# print(y,x)
		# time.sleep(100)
		# set sns context
		sns.set_context("notebook", font_scale=1.35, rc={"lines.linewidth": 1.2})
		sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
		#flatui = sns.color_palette("Dark2", 6)[::-1]
		flatui = sns.color_palette("Paired",12)
		flatui[-2] = (0.69411764705882351*1.4, 0.34901960784313724*1.4, 0.15686274509803921*1.4)
		#time.sleep(100)
		sns.set_palette(flatui)
		[methods.remove(i) for i in grouped] 
		methods.append("Random")
		methods.append("MostPopular")
		fig, ax = plt.subplots(figsize=(9,6))
		ls = '-'
		k = 0
		for g, group in df2.groupby("MML method"):
			df2b_ = df2b.loc[df2b['MML method'] == g]
			if g not in methods: continue
			x = np.array(group["Iteration index"])
			y = np.array(group[metric])
			x2 = np.array(df2b_["Iteration index"])
			y2 = np.array(df2b_[metric])
			yerror = np.array(group[metric+"std"])/4
			yerror2 = np.array(df2b_[metric+"std"])/2000
			ax.errorbar(x2+0.02*k, y2, yerr=yerror2, linestyle="--", marker='o', markersize=1)# label=g+"-nodrift")
			ax.errorbar(x+0.02*k, y, yerr=yerror, linestyle=ls, marker='o', markersize=7, label=g)
			
			k+=1
		ax.set_xticks(np.arange(len(x)) )
		ax.set_xlabel('Iterations')
		ax.set_ylabel(metric)
		ax.set_xlim(xmin=1)
		#ax.fill(xp, yp, "g",alpha=0.3, label = "Group A", joinstyle="round")
		ax.legend()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.show()

	# plot
	df2 = df.loc[df['MML method'] != "Control"]
	types = ["LongTail","LongTail","Unexpect","Unexpect","Unexpect","Business"]
	for i,metric in enumerate(["EPC","EFD","EPD","EILD", "ILD","Gini"]):
		g = sns.factorplot(x="Iteration index", y=metric, hue="MML method", data=df2, capsize=.2, palette=flatui, size=6, aspect=1, sharey = False, legend = True, dodge=True, legend_out=False)
		g.despine(left=True)
		plt.savefig(infolder + "/Analysis diversity "+types[i]+" - "+metric+".pdf")

def main(argv):
	helpText = 'analysis.py -f <infolder>'
	try:
		opts, args = getopt.getopt(argv,"hf:")
	except getopt.GetoptError:
		print(helpText)
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-f"):
			infolder = arg
	
	#analysis(infolder)
	#diversityAnalysis2(infolder)

	diversityAnalysis2(infolder,infolder+"-nodrift")
	

   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
#plt.show()