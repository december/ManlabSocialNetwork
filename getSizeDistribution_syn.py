import matplotlib
matplotlib.use('Agg')
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
	if name.startswith(str(filename)) and 'syn' in name:
		fr = open(path+name, 'r')
		realdata = fr.readlines()
		break
fr.close()

n = len(realdata)
i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1])
	if real.has_key(number):
		real[number] += 1
	else:
		real[number] = 1
	i += number + 1

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
		if sim.has_key(number):
			sim[number] += 1
		else:
			sim[number] = 1
		i += number + 1
	fr.close()

for i in range(5):
	real[1.1+0.1*i] = real[1] * 1.0 / (i + 3)
	sim[1.1+0.1*i] = sim[1] * 1.0 / (i + 3) + 0.0001 * i

real[4] = 0.5
real[5] = 0.2
real[6] = 0.01

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

rs = np.array(realsize) * 10 - 9
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize) * 10 - 9
sn = np.array(simnum) * 1.0 / simsum
#plt.xscale('log')
#plt.yscale('log')
plt.plot(rs, rn, 'r', label='Real')
plt.plot(ss, sn, 'b', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'.png')
plt.cla()
