import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def WienerIndex(gdic, j, k):
	wi = 0
	if gdic[j][1] < gdic[k][1]:
		while gdic[j][1] < gdic[k][1]:
			j = gdic[j][0]
			wi += 1
		if j == k:
			return wi
	if gdic[j][1] > gdic[k][1]:
		while gdic[j][1] > gdic[k][1]:
			k = gdic[k][0]
			wi += 1
		if j == k:
			return wi
	while j != k:
		j = gdic[j][0]
		k = gdic[k][0]
		wi += 2
	return wi
		
filename = int(sys.argv[1])
if filename < 0:
	single = False

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268

namelist = os.listdir(path)
real = {}
sim = {}
for name in namelist:
	if name.startswith(str(filename)):
		fr = open(path+name, 'r')
		realdata = fr.readlines()
		break
fr.close()

n = len(realdata)
i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	depdic = {}
	graphdic = {}
	keylist = list()
	for j in range(i+1, i+number):
		info = realdata[j].split('\t')
		keylist.append(info[0])
		graphdic[info[0]] = list()
		graphdic[info[0]].append(info[3])
		if info[3] == '-1':
			depdic[info[0]] = 0
		else:
			depdic[info[0]] = depdic[info[3]] + 1
		graphdic[info[0]].append(depdic[info[0]])
	wi = 0
	m = len(keylist)
	for j in range(m):
		for k in range(j+1, m):
			wi += WienerIndex(graphdic, keylist[j], keylist[k])
	if real.has_key(wi):
		real[wi] += 1
	else:
		real[wi] = 1
	i += number

namelist = os.listdir(prefix+str(filename)+'/')
m = len(namelist)
for name in namelist:
	fr = open(prefix+str(filename)+'/'+name, 'r')
	simdata = fr.readlines()
	n = len(simdata)
	i = 0
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1])
		depdic = {}
		graphdic = {}
		keylist = list()
		for j in range(i+1, i+number):
			info = realdata[j].split('\t')
			keylist.append(info[0])
			graphdic[info[0]] = list()
			graphdic[info[0]].append(info[3])
			if info[3] == '-1':
				depdic[info[0]] = 0
			else:
				depdic[info[0]] = depdic[info[3]] + 1
			graphdic[info[0]].append(depdic[info[0]])
		wi = 0
		m = len(keylist)
		for j in range(m):
			for k in range(j+1, m):
				wi += WienerIndex(graphdic, keylist[j], keylist[k])
		if sim.has_key(wi):
			sim[wi] += 1
		else:
			sim[wi] = 1
		i += number
	fr.close()

realsize = sorted(real.keys())
realnum = list()
for size in realsize:
	realnum.append(real[size])

simsize = sorted(sim.keys())
simnum = list()
for size in simsize:
	simnum.append(sim[size] * 1.0 / m)

realsum = sum(realnum)
simsum = sum(simnum)

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Wiener Index')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
plt.savefig(prefix+'WienerIndexDistribution/'+str(filename)+'.png')
plt.cla()
