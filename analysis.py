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


__author__ = 'Dimitrios Bountouridis'

df = pd.read_pickle("temp/history.pkl")
print(df)

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
flatui = sns.color_palette("husl", 8)


# sns.lmplot(x="inverseSalience", y="y", col="engine", hue="engine", data=df, col_wrap=2, ci=None, palette="muted", size=4, scatter_kws={"s": 50, "alpha": 1})
for g, group in df.groupby('itemid'):
	for i,group3 in group.groupby('engine'):
		for i,group2 in group3.groupby('userid'):
				total = np.array(group2).shape[0]
				if total>1:
					print("Potential problem:")
					print(group2)
					print(g,i,total)
					time.sleep(1)

# percentage of purchases that were recommended
D = []
for g, group in df.groupby('engine'):
	for i,group2 in group.groupby('iteration'):
		rec = np.sum(np.array(group2["wasRecommended"]))
		total = np.array(group2["wasRecommended"]).shape[0]
		D.append([g,i,rec/total])
print(D)
df_ = pd.DataFrame(D,columns=["engine","iteration","ratioOfRecommended"])
g = sns.factorplot(x="iteration", y="ratioOfRecommended", hue="engine", data=df_, capsize=.2, palette="YlGnBu_d", size=6, aspect=.75, sharey = False, legend = True)
g.despine(left=True)
# df.plot.box()
plt.show()

# purchases per type of: proximity or popular
g = sns.factorplot("ProximityOrPopular",col="engine", kind="count", data=df, capsize=.2, palette="YlGnBu_d", size=6, aspect=.75,  legend = True)
g.despine(left=True)
# df.plot.box()
plt.show()

# purchases per salience
g = sns.factorplot(y="inverseSalience",x="engine", kind="bar", data=df, capsize=.2, palette="YlGnBu_d", size=6, aspect=.75,  legend = True)
g.despine(left=True)
# df.plot.box()
plt.show()


# purchases per life period
g = sns.factorplot("lifespan",col="engine", kind="count", data=df, capsize=.2, palette="YlGnBu_d", size=6, aspect=.75,  legend = True)
g.despine(left=True)
# df.plot.box()
plt.show()

# purchases per topic category
g = sns.factorplot("class",col="engine", kind="count", data=df, capsize=.2, palette="YlGnBu_d", size=6, aspect=.75,  legend = True)
g.despine(left=True)
# df.plot.box()
plt.show()