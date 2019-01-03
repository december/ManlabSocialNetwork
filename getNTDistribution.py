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
			newy.append(tempy)
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
ts = 1321286400 #starting timestamps

namelist = os.listdir(path)
real_post = {}
real_rt = {}
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
delta = 100800
end = 702000
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	for j in range(i+1, i+number):
		info = realdata[j].split('\t')
		nt = float(info[2]) - ts - delta
		if nt <= 0 or nt >= end:
			continue
		if info[3] == '-1':
			if real_post.has_key(nt):
				real_post[nt] += 1
			else:
				real_post[nt] = 1
		else:
			if real_rt.has_key(nt):
				real_rt[nt] += 1
			else:
				real_rt[nt] = 1
	i += number

if single:
	namelist = os.listdir(prefix+str(filename)+'/')
	position = prefix+str(filename)+'/'
else:
	#namelist = os.listdir(prefix)
	#position = prefix
	namelist = os.listdir(prefix+'722911_twolevel_neighbor_cascades_simulation_10/')
	position = prefix+'722911_twolevel_neighbor_cascades_simulation_10/'	
cnt = 0
for name in namelist:
	if not name.endswith('.detail'):
		continue
	cnt = 1
	last_post = {}
	last_retweet = {}
	fr = open(position+name, 'r')
	simdata = fr.readlines()
	n = len(simdata)
	i = 0
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1]) + 1
		for j in range(i+1, i+number):
			info = simdata[j].split('\t')
			nt = float(info[2]) + 25200
			if nt <= 0 or nt >= end:
				continue	
			if info[3] == '-1':
				if sim_post.has_key(nt):
					sim_post[nt] += 1
				else:
					sim_post[nt] = 1
			else:
				if sim_rt.has_key(nt):
					sim_rt[nt] += 1
				else:
					sim_rt[nt] = 1
		i += number
	fr.close()

#print sim_post
#print sim_rt

start = 0

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


realsum = sum(realnum)
simsum = sum(simnum)

#binrx, binry = GetBin(1000, realsize, realcum) 
#rs = np.array(binrx)
#rn = np.array(binry) * 1.0 / realsum

rs = np.array(realsize[start:])
rn = np.array(realnum[start:])
rs, rn = GetBin(3600, rs, rn)
print rn
ss = np.array(simsize[start:])
sn = np.array(simnum[start:])
ss, sn = GetBin(3600, ss, sn)
#plt.xlim(xmin=1000)
#plt.xscale('log')
#plt.yscale('log')
plt.plot(rs, rn, 'r', label='Real', linewidth=2.5)
plt.plot(ss, sn, 'b', label='Our method', linewidth=2.5)
plt.xticks(fontsize=14)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.set_xticks([0, 20000, 40000, 60000, 80000])
plt.yticks(fontsize=14)
plt.xlabel(u'Natural time', fontsize=14)
plt.ylabel(u'Number of Messages', fontsize=14)
plt.title('Messages per Hour for Posting', fontsize=20)
plt.legend(loc='upper right', fontsize=20);
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_NTRoot.eps', dpi=1200)
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

realsum = sum(realnum)
simsum = sum(simnum)

#binrx, binry = GetBin(1000, realsize, realcum) 
#rs = np.array(binrx)
#rn = np.array(binry) * 1.0 / realsum

rs = np.array(realsize[start:])
rn = np.array(realnum[start:])
rs, rn = GetBin(3600, rs, rn)
ss = np.array(simsize[start:])
sn = np.array(simnum[start:])
ss, sn = GetBin(3600, ss, sn)
#plt.xlim(xmin=1000)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=14)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.set_xticks([0, 20000, 40000, 60000, 80000])
plt.yticks(fontsize=14)
plt.plot(rs, rn, 'r', label='Real', linewidth=2.5)
plt.plot(ss, sn, 'b', label='Our method', linewidth=2.5)
plt.xlabel(u'Natural time', fontsize=14)
plt.ylabel(u'Number of Messages', fontsize=14)
plt.title('Messages per Hour for Retweeting', fontsize=20)
plt.legend(loc='upper right', fontsize=20);
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_NTRetweet.eps',dpi=1200)
plt.cla()
