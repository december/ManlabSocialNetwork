import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def WienerIndex(gdic, j, k):
	wi = 0
	if gdic[j][1] > gdic[k][1]:
		while gdic[j][1] > gdic[k][1]:
			j = gdic[j][0]
			wi += 1
		if j == k:
			return wi
	if gdic[j][1] < gdic[k][1]:
		while gdic[j][1] < gdic[k][1]:
			k = gdic[k][0]
			wi += 1
		if j == k:
			return wi
	while j != k:
		j = gdic[j][0]
		k = gdic[k][0]
		wi += 2
	return wi

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

if single:
	namelist = os.listdir(prefix+str(filename)+'/')
	position = prefix+str(filename)+'/'
else:
	namelist = os.listdir(prefix)
	position = prefix
num = 0
for name in namelist:
	if not name.endswith('.detail'):
		continue
	fr = open(position+name, 'r')
	simdata = fr.readlines()
	n = len(simdata)
	i = 0
	num += 1
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1]) + 1
		depdic = {}
		graphdic = {}
		keylist = list()
		for j in range(i+1, i+number):
			info = simdata[j].split('\t')
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
	simnum.append(sim[size] * 1.0 / num)

realsum = sum(realnum)
simsum = sum(simnum)

realcum = [1]
n = len(realnum)
s = sum(realnum)
for i in range(n):
	s -= realnum[i]
	realcum.append(s)
simcum = [1]
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)
rs = np.array(realsize)
rn = np.array(realcum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simcum) * 1.0 / simsum
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Wiener Index')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'WienerDistribution/'+str(filename)+'_wiener_cum.png')
plt.cla()

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum

logmae = [0,0]
m = max(ss) + 1
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
logmae[0] = logmae[0] / m

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Wiener Index')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'WienerDistribution/'+str(filename)+'_wiener.png')
plt.cla()

binrx, binry = GetBin(1.4, realsize, realnum) 
binsx, binsy = GetBin(1.4, simsize, simnum)
rs = np.array(binrx)
rn = np.array(binry) * 1.0 / realsum
ss = np.array(binsx)
sn = np.array(binsy) * 1.0 / simsum

m = len(ss)
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
logmae[1] = logmae[1] / m
square = square / m
print logmae
print square

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Wiener Index')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'WienerDistribution/'+str(filename)+'_bin.png')
plt.cla()

binrx, binry = GetBin(1.4, realsize, realcum) 
binsx, binsy = GetBin(1.4, simsize, simcum)
rs = np.array(binrx)
rn = np.array(binry) * 1.0 / realsum
ss = np.array(binsx)
sn = np.array(binsy) * 1.0 / simsum

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Wiener Index')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'WienerDistribution/'+str(filename)+'wiener_cum_bin.png')
plt.cla()

