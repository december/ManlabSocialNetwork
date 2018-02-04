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

def isRepeat(tid, pr, author):
	pid = pr[tid]
	author1 = author[pr[pr[pid]]]
	author2 = author[pr[pid]]
	author3 = author[pid]
	author4 = author[tid]
	return (author1 == author3 and author2 == author4)

def GetBin(a, x, y):
	newx = list()
	newy = list()
	length = len(x)
	binnum = 1
	pos = 0
	s = 0
	tempy = 0
	tempx = 0
	while pos < length:
		s += a ** binnum
		tempx = s - a ** binnum / 2
		while x[pos] <= s:
			tempy += y[pos]
			pos += 1
			if pos >= length:
				break
		newx.append(tempx)
		newy.append(tempy)
		binnum += 1
		tempy = 0
	return newx, newy

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
	depdic = {}
	prdic = {}
	authordic = {}
	reflectdic = {}
	for j in range(i+1, i+number):
		info = realdata[j].split('\t')
		authordic[info[0]] = info[1]
		if info[3] == '-1':
			depdic[info[0]] = 1
		else:
			prdic[info[0]] = info[3]
			tempdep = depdic[info[3]] + 1
			depdic[info[0]] = tempdep
			if not reflectdic.has_key(info[3]) and tempdep >= 4:
				if isRepeat(info[0], prdic, authordic):
					depdic[prdic[info[0]]] -= 2
					depdic[info[0]] -= 2
					reflectdic[info[3]] = prdic[prdic[info[3]]]
					reflectdic[info[0]] = prdic[prdic[info[0]]]
	dep = max(depdic.values())
	if real.has_key(dep):
		real[dep] += 1
	else:
		real[dep] = 1
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
	m += 1
	n = len(simdata)
	i = 0
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1]) + 1
		depdic = {}
		for j in range(i+1, i+number):
			info = simdata[j].split('\t')
			if info[3] == '-1':
				depdic[info[0]] = 1
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
print realsize

simsize = sorted(sim.keys())
simnum = list()
for size in simsize:
	simnum.append(sim[size] * 1.0 / m)
print simsize

realsum = sum(realnum)
simsum = sum(simnum)
print realnum
print simnum

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
print rn
print sn

logmae = [0,0]
m = max(max(rs), max(ss))
pos1 = 0
pos2 = 0
cr = 0
cs = 0
for i in range(1, m):
	while pos1 < len(rs) and rs[pos1] < i:
		pos1 += 1
	while pos2 < len(ss) and ss[pos2] < i:
		pos2 += 1
	if pos1 < len(rs) and rs[pos1] == i:
		cr += rn[pos1]
	if pos2 < len(ss) and ss[pos2] == i:
		cs += sn[pos2]
	logmae[0] += abs(np.log(cr) - np.log(cs))

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Depth')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');
if not single:
	filename = 'all' 
plt.savefig(prefix+'DepthDistribution/'+str(filename)+'_depth.png')
plt.cla()

binrx, binry = GetBin(1.1, realsize, realnum) 
binsx, binsy = GetBin(1.1, simsize, simnum)
rs = np.array(binrx)
rn = np.array(binry) * 1.0 / realsum
ss = np.array(binsx)
sn = np.array(binsy) * 1.0 / simsum

m = max(len(rs), len(ss))
cr = 0
cs = 0
square = 0
for i in range(m):
	if i < len(rs):
		cr += rn[i]
	if i < len(ss):
		cs += sn[i]
	logmae[1] += abs(np.log(cr) - np.log(cs))
	square += abs(np.log(cr) - np.log(cs)) * 1.1 ** (i+1)
print logmae
print square

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Depth')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_bin.png')
plt.cla()
