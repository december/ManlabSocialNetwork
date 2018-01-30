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
if int(sys.argv[2]) == 0:
	prefix = '../../cascading_generation_model/simulation_notopic/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268

namelist = os.listdir(path)
real = {}
sim = {}
realdata = list()
for name in namelist:
	if single and name.startswith(str(filename) + '_'):
		fr = open(path+name, 'r')
		realdata = fr.readlines()
		fr.close()
		break
	if not single:
		fr = open(path+name, 'r')
		realdata.extend(fr.readlines())
		fr.close()

n = len(realdata)
i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	widdic = {}
	for j in range(i+1, i+number):
		info = realdata[j].split('\t')
		if widdic.has_key(info[3]):
			widdic[info[3]] += 1
		else:
			widdic[info[3]] = 1
	wid = max(widdic.values())
	if real.has_key(wid):
		real[wid] += 1
	else:
		real[wid] = 1
	i += number

if single:
	namelist = os.listdir(prefix+str(filename)+'/')
	position = prefix+str(filename)+'/'
else:
	namelist = os.listdir(prefix)
	position = prefix
m = 0
for name in namelist:
	if not name.endswith('.detail'):
		continue
	fr = open(position+name, 'r')
	simdata = fr.readlines()
	n = len(simdata)
	m += 1
	i = 0
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1]) + 1
		widdic = {}
		for j in range(i+1, i+number):
			info = simdata[j].split('\t')
			if widdic.has_key(info[3]):
				widdic[info[3]] += 1
			else:
				widdic[info[3]] = 1
		wid = max(widdic.values())
		if sim.has_key(wid):
			sim[wid] += 1
		else:
			sim[wid] = 1
		i += number
	fr.close()

realsize = sorted(real.keys())
realnum = list()
for size in realsize:
	realnum.append(real[size])
print realsize
print realsum

simsize = sorted(sim.keys())
simnum = list()
for size in simsize:
	simnum.append(sim[size] * 1.0 / m)
print simsize
print simsum

realsum = sum(realnum)
simsum = sum(simnum)

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Width')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right'); 
if not single:
	filename = 'all' 
plt.savefig(prefix+'WidthDistribution/'+str(filename)+'_width.png')
plt.cla()
