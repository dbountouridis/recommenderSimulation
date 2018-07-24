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
import glob
import sys, getopt

__author__ = 'Dimitrios Bountouridis'

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

	# print("Checking for issues...")
	# for g, group in df.groupby('Item'):
	# 	for i,group3 in group.groupby("MML method"):
	# 		for i,group2 in group3.groupby('User'):
	# 				total = np.array(group2).shape[0]
	# 				if total>1:
	# 					print("Potential problem:")
	# 					print(group2)
	# 					print(g,i,total)

	# set sns context
	sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	flatui = sns.color_palette("husl", 8)


	# plot
	df2 = df.loc[df['MML method'] != "Control"]
	g = sns.factorplot(x="Iteration index", y="EPC", hue="MML method", data=df2, capsize=.2, palette=flatui, size=6, aspect=0.7, sharey = False, legend = True, dodge=True, legend_out=False)
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis diversity - EPC.pdf")
	

	g = sns.factorplot(x="Iteration index", y="ILD", hue="MML method", data=df2, capsize=.2, palette=flatui, size=6, aspect=0.7, sharey = False, legend = True, dodge=True, legend_out=False)
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis diversity - ILD.pdf")

	g = sns.factorplot(x="Iteration index", y="Gini", hue="MML method", data=df2, capsize=.2, palette=flatui, size=6, aspect=0.7, sharey = False, legend = True, dodge=True, legend_out=False)
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis diversity - Gini.pdf")





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

	# print("Checking for issues...")
	# for g, group in df.groupby('Item'):
	# 	for i,group3 in group.groupby("MML method"):
	# 		for i,group2 in group3.groupby('User'):
	# 				total = np.array(group2).shape[0]
	# 				if total>1:
	# 					print("Potential problem:")
	# 					print(group2)
	# 					print(g,i,total)

	# set sns context
	sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	flatui = sns.color_palette("husl", 8)

	# percentage of purchases that were recommended
	D = []
	for g, group in df.groupby("MML method"):	
		for i,group3 in group.groupby("Iteration index"):
			f = []
			for t,group2 in group3.groupby("Class/Topic"):
				total = np.array(group2).shape[0]
				f.append(total)
			D.append([g]+f)
	# #print(D,["MML method"]+[t for t,group in df.groupby("Class/Topic")])
	# df_ = pd.DataFrame(D,columns=["MML method"]+[t for t,group in df.groupby("Class/Topic")] )
	# print(df_)
	# for g, group in df_.groupby("MML method"):
	# 	print(group)
	# 	group.plot.bar(stacked=True);
	# 	plt.show()


	# # plot
	# cmaps= ['Blues','Reds','Greens','Oranges','Greys']
	# t = ["entertainment","business","sport","politics","tech"]
	# colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(t))]
	# g = sns.factorplot(x="Iteration index", y="Counts", hue="Class/Topic", col="MML method", data=df_, capsize=.2, palette=colors, size=3, aspect=2, sharey = True, legend = True, dodge=True, legend_out=False, hue_order = t)
	# g.despine(left=True)
	# plt.savefig(infolder + "/Analysis - Selected articles that were recommended.pdf")
	# #plt.show()

	g = sns.factorplot(y="Agreement between deterministic and stochastic choice",x="MML method", kind="bar", data=df, capsize=.2, palette=flatui, size=4, aspect=1.5,  legend = True, legend_out=False,dodge=True, orient = "v")
	plt.savefig(infolder + "/Analysis - Agreement between deterministic and stochastic choice.pdf")
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
	
	analysis(infolder)
	diversityAnalysis(infolder)
	

   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
#plt.show()