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

def fakeSim(engine,iterations,outfolder,totalNumberOfUsers,totalNumberOfItems):
	print("======== Engine "+engine+" ==========")
	for i,iter_ in enumerate(iterations):
		print("* Epoch:",iter_,", index:",i)
		print("  !random stuff here, not important! ")
		d = {"Current recommender":engine,"Epoch":i,"Completed":(i+1)/len(iterations),"Available items": int(random.random()*500),"Average Awareness": random.random()*40, "Some other attribute": random.random(),"Users Position":[(random.random(),random.random()) for i in range(totalNumberOfUsers)],"Items Position":[(random.random(),random.random()) for i in range(totalNumberOfUsers)]}
		with open(outfolder+'/'+str(engine)+'-data.json', 'w') as outfile:
			json.dump(d, outfile)
		time.sleep(1)

def main(argv):
	helpText = 'simulationClass.py  -i <iterations> -s <seed> -u <totalusers> -t <totalitems> -d <deltasalience> -r <recommenders> -f <outfolder>'
	try:
		opts, args = getopt.getopt(argv,"hi:s:u:d:r:f:t:")
	except getopt.GetoptError:
		print(helpText)
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print(helpText)
			sys.exit()
		elif opt in ("-u"):
			totalNumberOfUsers = int(arg)
		elif opt in ("-t"):
			totalNumberOfItems = int(arg)
		elif opt in ("-i"):
			totalNumberOfIterations = int(arg)
		elif opt in ("-s"):
			seed = arg
		elif opt in ("-d"):
			delta = arg
		elif opt in ("-f"):
			outfolder = arg
		elif opt in ("-r"):
			if "," in arg: recommenders = arg.split(",") 
			else: recommenders = [arg]
	
	print("Initialize simulation class...")
	print("Create simulation instance...")
	print("Plotting users/items in 2d space...")
	print("Run Control period...")
	fakeSim("Control",[i for i in range(int(totalNumberOfIterations/2))],outfolder, totalNumberOfUsers,totalNumberOfItems)
	print("Plotting...")	
	print("Run Recommenders....")
	for rec in recommenders:
		fakeSim(rec,[i for i in range(int(totalNumberOfIterations/2),totalNumberOfIterations)],outfolder,totalNumberOfUsers,totalNumberOfItems)
	print("Finished!")

if __name__ == "__main__":
   main(sys.argv[1:])  