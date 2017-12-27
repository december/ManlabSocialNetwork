import matplotlib.pyplot as plt
import numpy as np
import sys
import os

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
	for j in range(i+1, i+number):
		info = realdata[j].split('\t')
		if info[3] == '-1':
			depdic[info[0]] = 0
		else:
			depdic[info[0]] = depdic[info[3]] + 1
	dep = max(depdic.values())
	if real.has_key(dep):
		real[dep] += 1
	else:
		real[dep] = 1
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
		for j in range(i+1, i+number):
			info = realdata[j].split('\t')
			if info[3] == '-1':
				depdic[info[0]] = 0
			else:
				depdic[info[0]] = depdic[info[3]] + 1
		dep = max(depdic.values())
		if sim.has_key(dep):
			sim[dep] += 1
		else:
			sim[dep] = 1
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
plt.xlabel(u'Depth')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'.png')
plt.cla()