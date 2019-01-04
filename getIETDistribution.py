import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
sns.set()
sns.set_style('white')

filename = int(sys.argv[1])
if filename < 0:
	single = False

def GetBin(a, x, y):
	newx = list()
	newy = list()
	length = len(x)
	binnum = 1
	pos = 0
	s = 0
	tempy = 0
	tempx = 0
	tempz = 0
	while pos < length:
		s += a
		tempx = s - a / 2
		while x[pos] <= s:
			tempy += y[pos]
			tempz += 1
			pos += 1
			if pos >= length:
				break
		if tempy > 0:
			newx.append(tempx)
			newy.append(tempy / tempz)
		binnum += 1
		tempy = 0
		tempz = 0
	return newx, newy


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
base = {}
base_post = {}
base_rt = {}
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
	#namelist = os.listdir(prefix+'722911_twolevel_neighbor_cascades_simulation_10/')
	#position = prefix+'722911_twolevel_neighbor_cascades_simulation_10/'	
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

cnt = 1
if single:
	namelist = os.listdir(prefix+str(filename)+'/')
	position = prefix+str(filename)+'/'
else:
	namelist = os.listdir(prefix)
	position = prefix
	#namelist = os.listdir(prefix+'722911_twolevel_neighbor_cascades_simulation_10/')
	#position = prefix+'722911_twolevel_neighbor_cascades_simulation_10/'	
for name in namelist:
	if not name.endswith('.detail'):
		continue
	last_post = {}
	last_retweet = {}
	fr = open(position+name, 'r')
	basedata = fr.readlines()
	n = len(simdata)
	i = 0
	while i < n:
		temp = basedata[i].split('\t')
		number = int(temp[1]) + 1
		timedic = {}
		newtemp = basedata[i+1].split('\t')
		if not last_post.has_key(newtemp[1]):
			last_post[newtemp[1]] = list()
		last_post[newtemp[1]].append(int(float(newtemp[2])))
		for j in range(i+1, i+number):
			info = basedata[j].split('\t')
			tp = int(float(info[2]))
			timedic[info[0]] = tp
			if info[3] != '-1':
				if not last_retweet.has_key(info[1]):
					last_retweet[info[1]] = list()
				last_retweet[info[1]].append(tp)
				timedic[info[0]] = tp
				tempiet = tp - timedic[info[3]]
				if base.has_key(tempiet):
					base[tempiet] += 1
				else:
					base[tempiet] = 1
		i += number
	for item in last_post:
		last_post[item].sort()
		m = len(last_post[item])
		for i in range(m-1):
			tempiet = last_post[item][i+1] - last_post[item][i]
			if base_post.has_key(tempiet):
				base_post[tempiet] += 1
			else:
				base_post[tempiet] = 1
	for item in last_retweet:
		last_retweet[item].sort()
		m = len(last_retweet[item])
		for i in range(m-1):
			tempiet = last_retweet[item][i+1] - last_retweet[item][i]
			if base_rt.has_key(tempiet):
				base_rt[tempiet] += 1
			else:
				base_rt[tempiet] = 1
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
	simnum.append(sim[size] * 1.0 / cnt)
simcum = list()
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)

basesize = sorted(base.keys())
basenum = list()
for size in basesize:
	basenum.append(base[size] * 1.0 / cnt)
basecum = list()
n = len(basenum)
s = sum(basenum)
for i in range(n):
	s -= basenum[i]
	basecum.append(s)	

realsum = sum(realnum)
simsum = sum(simnum)
basesum = sum(basenum)

start = 0

rs = np.array(realsize[start:])
rn = np.array(realcum[start:]) * 1.0 / realsum
#rs, rn = GetBin(10, rs, rn)
ss = np.array(simsize[start:])
sn = np.array(simcum[start:]) * 1.0 / simsum
#plt.xlim(xmin=1000)
bs = np.array(basesize[start:])
bn = np.array(basecum[start:]) * 1.0 / basesum
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real', linewidth=2.5)
plt.plot(ss, sn, 'b', label='Our method', linewidth=2.5)
plt.plot(bs, bn, 'k', label='Poisson', linewidth=2.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(u'Response Time', fontsize=14)
plt.ylabel(u'Distribution', fontsize=14)
plt.legend(loc='lower left', fontsize=20);
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_RT.eps', dpi=1200)
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
	simnum.append(sim_post[size] * 1.0 / cnt)
simcum = list()
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)

basesize = sorted(base_post.keys())
basenum = list()
for size in basesize:
	basenum.append(base_post[size] * 1.0 / cnt)
basecum = list()
n = len(basenum)
s = sum(basenum)
for i in range(n):
	s -= basenum[i]
	basecum.append(s)	

realsum = sum(realnum)
simsum = sum(simnum)
basesum = sum(basenum)

#binrx, binry = GetBin(1000, realsize, realcum) 
#rs = np.array(binrx)
#rn = np.array(binry) * 1.0 / realsum

rs = np.array(realsize[start:])
rn = np.array(realcum[start:]) * 1.0 / realsum
rs, rn = GetBin(20000, rs, rn)
ss = np.array(simsize[start:])
sn = np.array(simcum[start:]) * 1.0 / simsum
bs = np.array(basesize[start:])
bn = np.array(basecum[start:]) * 1.0 / basesum
#plt.xlim(xmin=1000)
#plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real', linewidth=2.5)
plt.plot(ss, sn, 'b', label='Our method', linewidth=2.5)
plt.plot(bs, bn, 'k', label='Poisson', linewidth=2.5)
plt.xticks(fontsize=14)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.set_xticks([0, 20000, 40000, 60000, 80000])
plt.yticks(fontsize=14)
plt.xlabel(u'Interevent time', fontsize=14)
plt.ylabel(u'CDF', fontsize=14)
plt.title('IET Distribution for Posting Roots', fontsize=20)
plt.legend(loc='upper right', fontsize=20);
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_IETRoot.eps', dpi=1200)
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
	simnum.append(sim_rt[size] * 1.0 / cnt)
simcum = list()
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)

basesize = sorted(base_rt.keys())
basenum = list()
for size in basesize:
	basenum.append(base_rt[size] * 1.0 / cnt)
basecum = list()
n = len(basenum)
s = sum(basenum)
for i in range(n):
	s -= basenum[i]
	basecum.append(s)	

realsum = sum(realnum)
simsum = sum(simnum)
basesum = sum(basenum)

#binrx, binry = GetBin(1000, realsize, realcum) 
#rs = np.array(binrx)
#rn = np.array(binry) * 1.0 / realsum

rs = np.array(realsize[start:])
rn = np.array(realcum[start:]) * 1.0 / realsum
rs, rn = GetBin(20000, rs, rn)
ss = np.array(simsize[start:])
sn = np.array(simcum[start:]) * 1.0 / simsum
bs = np.array(basesize[start:])
bn = np.array(basecum[start:]) * 1.0 / basesum
#plt.xlim(xmin=1000)
#plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=14)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.set_xticks([0, 20000, 40000, 60000, 80000])
plt.yticks(fontsize=14)
plt.plot(rs, rn, 'ro', label='Real', linewidth=2.5)
plt.plot(ss, sn, 'b', label='Our method', linewidth=2.5)
plt.plot(bs, bn, 'k', label='Poisson', linewidth=2.5)
plt.xlabel(u'Interevent time', fontsize=14)
plt.ylabel(u'CDF', fontsize=14)
plt.title('IET Distribution for Retweeting', fontsize=20)
plt.legend(loc='upper right', fontsize=20);
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_IETRetweet.eps',dpi=1200)
plt.cla()
