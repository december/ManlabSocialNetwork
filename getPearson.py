import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
import os

realdic = {} #from id to cascade 0-1 list
relation = {} #from id to follower id
authordic = {} #from tweet id to author id
pr_sim = 0
pr_random = 0
pr_sim_sum = 0
rr_sim = 0
rr_random = 0
rr_sim_sum = 0
all_sim = 0
all_random = 0

pr_real = 0
pr_rd = 0
pr_real_sum = 0
rr_real = 0
rr_rd = 0
rr_real_sum = 0
all_real = 0
all_rd = 0

def calcMean(x, y):  
   sum_x = sum(x)  
   sum_y = sum(y)  
   n = len(x)  
   x_mean = float(sum_x+0.0) / n  
   y_mean = float(sum_y+0.0) / n  
   return x_mean, y_mean  

def calcPearson(x, y):
	if not realdic.has_key(x):
		return 0
	if not realdic.has_key(y):
		return 0
	x = realdic[x]
	y = realdic[y]
	#print str(len(x)) + ' ' + str(len(y)) 
	x_mean, y_mean = calcMean(x,y)
	n = len(x)
	sumTop = 0.0  
	sumBottom = 0.0  
	x_pow = 0.0  
	y_pow = 0.0  
	for i in range(n):
		sumTop += (x[i] - x_mean) * (y[i] - y_mean)  
	for i in range(n):
		x_pow += math.pow(x[i]-x_mean, 2)  
	for i in range(n):
		y_pow += math.pow(y[i]-y_mean, 2)  
	sumBottom = math.sqrt(x_pow * y_pow) 
	if sumBottom == 0:
		return 0 
	p = sumTop / sumBottom  
	return abs(p)  

def randomSelect(person):
	friend = random.choice(relation[person])
	return calcPearson(person, friend)

def chooseTwo(person):
	friend = relation[person]
	f1 = random.choice(friend)
	friend.remove(f1)
	f2 = random.choice(friend)
	friend.append(f1)
	return calcPearson(f1, f2)

filename = int(sys.argv[1])
single = True
if filename < 0:
	single = False

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268
relation_prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
if single:
	relation_prefix += 'single_user_parameter/'


namelist = os.listdir(path)
real = {}
sim = {}
for name in namelist:
	if name.startswith(str(filename) + '_'):
		fr = open(path+name, 'r')
		realdata = fr.readlines()
		break
fr.close()

n = len(realdata)
i = 0
cnt = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	rdic = {}
	for j in range(i+1, i+number):
		data = realdata[j].split('\t')
		if rdic.has_key(data[1]):
			continue
		if not realdic.has_key(data[1]):
			realdic[data[1]] = list()
			for k in range(cnt):
				realdic[data[1]].append(0)
		realdic[data[1]].append(1)
		rdic[data[1]] = 1
	for key in realdic:
		if not rdic.has_key(key):
			realdic[key].append(0)
	cnt += 1
	i += number

#print cnt

fr = open(relation_prefix+'pi_Poisson_'+str(filename)+suffix, 'r')
pilist = fr.readlines()
enum = len(pilist)
for i in range(enum):
	temp = pilist[i].split('\t')
	if not relation.has_key(temp[0]):
		relation[temp[0]] = list()
	relation[temp[0]].append(temp[1])
fr.close()

i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	rtdic = {}
	for j in range(i+1, i+number):
		data = realdata[j][:-1].split('\t')
		#print data
		if data[3] != '-1':
			authordic[data[3]] = data[4]
			if not rtdic.has_key(data[3]):
				rtdic[data[3]] = {}
			if not rtdic[data[3]].has_key(data[1]):
				rtdic[data[3]][data[1]] = 1
			else:
				rtdic[data[3]][data[1]] += 1
			pr_real += calcPearson(data[1], data[4])
			pr_rd += randomSelect(data[4])
			pr_real_sum += 1
		for key in rtdic:
			if len(rtdic[key]) <= 1:
				continue
			keylist = rtdic[key].keys()
			m = len(keylist)
			for j in range(m):
				for k in range(j+1, m):
					rr_real += calcPearson(rtdic[key][keylist[j]], rtdic[key][keylist[k]])
					rr_rd += chooseTwo(authordic[key])
					rr_real_sum += 1
	i += number

namelist = os.listdir(prefix+str(filename)+'/')
cnt = 0
for name in namelist:
	fr = open(prefix+str(filename)+'/'+name, 'r')
	simdata = fr.readlines()
	n = len(simdata)
	i = 0
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1]) + 1
		rtdic = {}
		for j in range(i+1, i+number):
			data = simdata[j][:-1].split('\t')
			if data[3] != '-1':
				authordic[data[3]] = data[4]
				if not rtdic.has_key(data[3]):
					rtdic[data[3]] = {}
				if not rtdic[data[3]].has_key(data[1]):
					rtdic[data[3]][data[1]] = 1
				else:
					rtdic[data[3]][data[1]] += 1
				pr_sim += calcPearson(data[1], data[4])
				pr_random += randomSelect(data[4])
				pr_sim_sum += 1
		for key in rtdic:
			if len(rtdic[key]) <= 1:
				continue
			keylist = rtdic[key].keys()
			m = len(keylist)
			#print rtdic[key]
			#print relation[authordic[key]]
			#print authordic[key]
			for j in range(m):
				for k in range(j+1, m):
					rr_sim += calcPearson(rtdic[key][keylist[j]], rtdic[key][keylist[k]])
					rr_random += chooseTwo(authordic[key])
					rr_sim_sum += 1
		i += number
	cnt += 1
	print cnt
	fr.close()

all_real = (rr_real + pr_real) / (rr_real_sum + pr_real_sum)
all_rd = (rr_rd + pr_rd) / (rr_real_sum + pr_real_sum)
all_sim = (rr_sim + pr_sim) / (rr_sim_sum + pr_sim_sum)
all_random = (rr_random + pr_random) / (rr_sim_sum + pr_sim_sum)
rr_real = rr_real / rr_real_sum
rr_rd = rr_rd / rr_real_sum
pr_real = pr_real / pr_real_sum
pr_rd = pr_rd / pr_real_sum
pr_sim = pr_sim / pr_sim_sum
pr_random = pr_random / pr_sim_sum
rr_sim = rr_sim / rr_sim_sum
rr_random = rr_random / rr_sim_sum

y_sim = [rr_real, pr_real, all_real, rr_sim, pr_sim, all_sim]
y_random = [rr_rd, pr_rd, all_rd, rr_random, pr_random, all_random]
x = np.arange(6)

plt.bar(x , y_sim, width=0.3 , color='y')
plt.bar(x+0.3, y_random, width=0.3 , color='b')
plt.savefig(prefix+'Pearson/'+str(filename)+'.png')
