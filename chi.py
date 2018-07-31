from scipy.stats import chi2
from scipy import spatial
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import random
import bisect
import collections
import seaborn as sns

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]



fig, ax = plt.subplots(1, 1)
#Calculate a few first moments:

categories = ["entertainment","business","sport","politics","tech"]
categoriesSalience = [0.05,0.07,0.03,0.85,0.01] # arbitrary assigned
items = 3000
weights=[1/len(categories) for i in range(len(categories))]
population = categories
counts = collections.defaultdict(int)
for i in range(items): counts[choice(population, weights)] += 1
print(counts)


df = 2
mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), items)
rv = chi2(df)

# first sanity check
D = []
for j in range(10):
	for i in rv.pdf(x[:5]):
		c = choice(population, categoriesSalience)
		while counts[c]<=0:
			c = choice(population, categoriesSalience)
		counts[c]-=1
		D.append(c)
for tupl in collections.Counter(D).most_common():
	print(tupl[0],tupl[1]/len(D))

# second, histogram
D = {}
for c in categories: D.update({c:[]})

for j in range(10):
	counts = collections.defaultdict(int)
	for i in range(items): counts[choice(population, weights)] += 1
	for i in rv.pdf(x):
		c = choice(population, categoriesSalience)
		while counts[c]<=0:
			c = choice(population, categoriesSalience)
		counts[c]-=1
		D[c].append(i/0.5)

min_= np.min([len(D[i]) for i in D.keys()])
print(min_)

x = []
for k in D.keys():
	x.append(D[k][:min_])
print(np.array(x).T)

# set sns context
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.2,'text.usetex' : True})
sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
flatui = sns.color_palette("husl", 8)
#fig, ax = plt.subplots()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
ax0= axes

cmaps= ['Blues','Reds','Greens','Oranges','Greys']
t = ["entertainment","business","sport","politics","tech"]
colors = [sns.color_palette(cmaps[i])[-2] for i in range(len(t))]
ax0.hist(x, 10, histtype='bar',stacked=True, color=colors,label=categories)
ax0.legend(prop={'size': 15})
ax0.set_xlabel("$z^0$")
ax0.set_ylabel("counts")
plt.show()




