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

	# plot
	df = df.loc[df['MML method'] != "Control"]
	df.columns = ["EPC", "ILD", "Gini", "EFD", "EPD", "EILD","Day","News Website"]
	days = ["Mon", "Tue", "Wed","Thu","Fri","Sat","Sun"]
	for i in range(0,7):
		df.ix[df["Day"]==i, ["Day"]] = days[i]

	allw = list(set(np.array(df["News Website"]).tolist()))
	NW = ["A", "B", "C", "D"]
	print(allw)
	for i, wp in enumerate(allw):
		df.ix[df["News Website"]==wp, ["News Website"]] = NW[i]

	df['EPC'] = df['EPC'].apply(lambda x: x*100)
	df['EILD'] = df['EILD'].apply(lambda x: x*1000)
	print(df)

	df.columns = ['Undiscovered articles', 'ILD', "Gini", "EFD", "EPD","Unexpected articles","Day","News Website"]

		# set sns context
	sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1.2})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	flatui = sns.color_palette("husl", 8)

	types = ["LongTail","LongTail","Unexpect","Unexpect","Unexpect","Business"]
	for i,metric in enumerate(['Undiscovered articles',"Unexpected articles"]):
		g = sns.factorplot(x="Day", y=metric, hue="News Website", data=df, capsize=.2, palette=flatui, size=8, aspect=1.2, sharey = False, legend = True, dodge=True, legend_out= True, order = days, hue_order = NW)
		g.despine(left=True)
		plt.savefig(infolder + "/Analysis diversity "+types[i]+" - "+metric+".png")


		g = sns.factorplot(x="News Website",y=metric, kind="bar", data=df, capsize=.2,  palette=flatui, size=7, aspect=1.8,  legend = True, dodge=True, legend_out= True, order = ["A", "B", "C", "D"])
		g.despine(left=True)
		plt.savefig(infolder + "/Analysis diversity "+types[i]+" - "+metric+"2.png")






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


	# set sns context
	sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1.2})
	sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	flatui = sns.color_palette("husl", 8)


	df = df.loc[df['MML method'] != "Control"]
	df = df.drop(columns=["Item Prominence", 'User', 'Was Recommended',"Agreement between deterministic and stochastic choice","Item has been recommended before"])
	df.columns = ['Day', 'News Website', "Item", "Days after event", "Topic"]

	allw = list(set(np.array(df["News Website"]).tolist()))
	NW = ["A", "B", "C", "D", "E"]
	print(allw)
	for i, wp in enumerate(allw):
		df.ix[df["News Website"]==wp, ["News Website"]] = NW[i]
	
	days = ["Mon", "Tue", "Wed","Thu","Fri","Sat","Sun"]
	for i in range(0,7):
		df.ix[df["Day"]==i, ["Day"]] = days[i]

	df["Days after event"] = df["Days after event"].astype(int)

	g = sns.factorplot("Days after event",col="News Website", kind="count", data=df, capsize=.2, palette=sns.color_palette("BuGn_r",15), size=6, aspect=.75,  legend = True, legend_out=False,dodge=True, col_order = ["A", "B", "C", "D"])
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis - Item Age.png")
	#plt.show()

	cmaps= ['Blues','Reds','Greens','Oranges','Greys']
	t = ["entertainment","business","sport","politics","tech"]
	colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(t))]

	g = sns.factorplot(x="News Website", hue="Topic",kind="count", data=df, capsize=.2, palette=colors, size=7, aspect=1.8,  legend = True, dodge=True, legend_out=False, hue_order=t, order = ["A", "B", "C", "D"])
	g.despine(left=True)
	plt.savefig(infolder + "/Analysis - Class-Topic.png")
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