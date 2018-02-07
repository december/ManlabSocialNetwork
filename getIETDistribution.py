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
real_post = {}
real_rt = {}
sim = {}
sim_post = {}
sim_rt = {}
last_post = {}
last_retweet = {}
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
	timedic = {}
	newtemp = realdata[i+1].split('\t')
	if not last_post.has_key(newtemp[1]):
		last_post[newtemp[1]] = list()
	last_post[newtemp[1]].append(int(float(newtemp[2])))
	for j in range(i+1, i+number):
		info = realdata[j].split('\t')
		tp = int(float(info[2]))
		timedic[info[0]] = tp
		if info[3] != '-1':
			if not last_retweet.has_key(info[1]):
				last_retweet[info[1]] = list()
			last_retweet[info[1]].append(tp)
			timedic[info[0]] = tp
			tempiet = tp - timedic[info[3]]
			if real.has_key(tempiet):
				real[tempiet] += 1
			else:
				real[tempiet] = 1
	i += number
for item in last_post:
	last_post[item].sort()
	m = len(last_post[item])
	for i in range(m-1):
		tempiet = last_post[item][i+1] - last_post[item][i]
		if real_post.has_key(tempiet):
			real_post[tempiet] += 1
		else:
			real_post[tempiet] = 1
for item in last_retweet:
	last_retweet[item].sort()
	m = len(last_retweet[item])
	for i in range(m-1):
		tempiet = last_retweet[item][i+1] - last_retweet[item][i]
		if real_rt.has_key(tempiet):
			real_rt[tempiet] += 1
		else:
			real_rt[tempiet] = 1

if single:
	namelist = os.listdir(prefix+str(filename)+'/')
	position = prefix+str(filename)+'/'
else:
	namelist = os.listdir(prefix)
	position = prefix
m = len(namelist)
for name in namelist:
	if not name.endswith('.detail'):
		continue
	last_post = {}
	last_retweet = {}
	fr = open(position+name, 'r')
	simdata = fr.readlines()
	n = len(simdata)
	i = 0
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1]) + 1
		timedic = {}
		newtemp = simdata[i+1].split('\t')
		if not last_post.has_key(newtemp[1]):
			last_post[newtemp[1]] = list()
		last_post[newtemp[1]].append(int(float(newtemp[2])))
		for j in range(i+1, i+number):
			info = simdata[j].split('\t')
			tp = int(float(info[2]))
			timedic[info[0]] = tp
			if info[3] != '-1':
				if not last_retweet.has_key(info[1]):
					last_retweet[info[1]] = list()
				last_retweet[info[1]].append(tp)
				timedic[info[0]] = tp
				tempiet = tp - timedic[info[3]]
				if sim.has_key(tempiet):
					sim[tempiet] += 1
				else:
					sim[tempiet] = 1
		i += number
	for item in last_post:
		last_post[item].sort()
		m = len(last_post[item])
		for i in range(m-1):
			tempiet = last_post[item][i+1] - last_post[item][i]
			if sim_post.has_key(tempiet):
				sim_post[tempiet] += 1
			else:
				sim_post[tempiet] = 1
	for item in last_retweet:
		last_retweet[item].sort()
		m = len(last_retweet[item])
		for i in range(m-1):
			tempiet = last_retweet[item][i+1] - last_retweet[item][i]
			if sim_rt.has_key(tempiet):
				sim_rt[tempiet] += 1
			else:
				sim_rt[tempiet] = 1
		fr.close()

realsize = sorted(real.keys())
realnum = list()
for size in realsize:
	realnum.append(real[size])
realcum = list()
n = len(realnum)
s = sum(realnum)
for i in range(n):
	s -= realnum[i]
	realcum.append(s)

simsize = sorted(sim.keys())
simnum = list()
for size in simsize:
	simnum.append(sim[size] * 1.0 / m)
simcum = list()
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)

realsum = sum(realnum)
simsum = sum(simnum)

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'r', label='Real')
plt.plot(ss, sn, 'b', label='Sim')
plt.xlabel(u'Response Time')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_RT.png')
plt.cla()

realsize = sorted(real_post.keys())
realnum = list()
for size in realsize:
	realnum.append(real_post[size])
realcum = list()
n = len(realnum)
s = sum(realnum)
for i in range(n):
	s -= realnum[i]
	realcum.append(s)

simsize = sorted(sim_post.keys())
simnum = list()
for size in simsize:
	simnum.append(sim_post[size] * 1.0 / m)
simcum = list()
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)


realsum = sum(realnum)
simsum = sum(simnum)

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'r', label='Real')
plt.plot(ss, sn, 'b', label='Sim')
plt.xlabel(u'IET for Root')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_IETRoot.png')
plt.cla()

realsize = sorted(real_rt.keys())
realnum = list()
for size in realsize:
	realnum.append(real_rt[size])
realcum = list()
n = len(realnum)
s = sum(realnum)
for i in range(n):
	s -= realnum[i]
	realcum.append(s)

simsize = sorted(sim_rt.keys())
simnum = list()
for size in simsize:
	simnum.append(sim_rt[size] * 1.0 / m)
simcum = list()
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)

realsum = sum(realnum)
simsum = sum(simnum)

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'r', label='Real')
plt.plot(ss, sn, 'b', label='Sim')
plt.xlabel(u'IET for Retweet')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_IETRetweet.png')
plt.cla()
