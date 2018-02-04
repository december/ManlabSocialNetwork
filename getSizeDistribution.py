import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

single = True
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
	fr = open(path+name, 'r')
	realdata.extend(fr.readlines())
	fr.close()

n = len(realdata)
i = 0
bigreal = list()
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1])
	a = int(realdata[i+1].split('\t')[1])
	if a == filename or not single:
		if number >= 50:
			for j in range(i, i+number+1):
				bigreal.append(realdata[j])
		if real.has_key(number):
			real[number] += 1
		else:
			real[number] = 1
	i += number + 1

namelist = os.listdir(prefix)
position = prefix
m = 0
bigsim = list()
for name in namelist:
	if not name.endswith('.detail'):
		continue
	m += 1
	fr = open(position+name, 'r')
	simdata = fr.readlines()
	n = len(simdata)
	i = 0
	while i < n:
		temp = simdata[i].split('\t')
		number = int(temp[1])
		a = int(simdata[i+1].split('\t')[1])
		if a == filename or not single:
			if number >= 50:
				for j in range(i, i+number+1):
					bigsim.append(simdata[j])
			if sim.has_key(number):
				sim[number] += 1
			else:
				sim[number] = 1
		i += number + 1
	fr.close()

realsize = sorted(real.keys())
print realsize
realnum = list()
realcum = list()
for size in realsize:
	realnum.append(real[size])
n = len(realnum)
s = sum(realnum)
for i in range(n):
	s -= realnum[i]
	realcum.append(s)

simsize = sorted(sim.keys())
print simsize
simnum = list()
simcum = list()
for size in simsize:
	simnum.append(sim[size] * 1.0 / m)
n = len(simnum)
s = sum(simnum)
for i in range(n):
	s -= simnum[i]
	simcum.append(s)

realsum = sum(realnum)
simsum = sum(simnum)
print simcum
print realcum

logmae = [0,0,0,0]
rs = np.array(realsize)
rn = np.array(realcum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simcum) * 1.0 / simsum

m = max(max(rs), max(ss))
pos1 = 0
pos2 = 0
for i in range(m):
	temp = 0
	while pos1 < len(rs) and rs[pos1] < i:
		pos1 += 1
	while pos2 < len(ss) and ss[pos2] < i:
		pos2 += 1
	if pos1 < len(rs) and rs[pos1] == i:
		temp += rn[pos1]
	if pos2 < len(ss) and ss[pos2] == i:
		temp -= sn[pos2]
	logmae[0] += np.log(abs(temp))

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_cum.png')
plt.cla()

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum

m = max(max(rs), max(ss))
pos1 = 0
pos2 = 0
for i in range(m):
	temp = 0
	while pos1 < len(rs) and rs[pos1] < i:
		pos1 += 1
	while pos2 < len(ss) and ss[pos2] < i:
		pos2 += 1
	if rs[pos1] == i:
		temp += rn[pos1]
	if ss[pos2] == i:
		temp -= sn[pos2]
	logmae[1] += np.log(abs(temp))

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_num.png')
plt.cla()

square = [0, 0]
binrx, binry = GetBin(1.1, realsize, realcum) 
binsx, binsy = GetBin(1.1, simsize, simcum)
rs = np.array(binrx)
rn = np.array(binry) * 1.0 / realsum
ss = np.array(binsx)
sn = np.array(binsy) * 1.0 / simsum

m = max(len(rs), len(ss))
for i in range(m):
	temp = 0
	if i < len(rs):
		temp += rn[i]
	if i < len(ss):
		temp -= sn[i]
	logmae[2] += np.log(abs(temp))
	square[0] += np.log(abs(temp) * 1.1 ** i)

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_cum_bin.png')
plt.cla()

binrx, binry = GetBin(1.1, realsize, realnum) 
binsx, binsy = GetBin(1.1, simsize, simnum)
rs = np.array(binrx)
rn = np.array(binry) * 1.0 / realsum
ss = np.array(binsx)
sn = np.array(binsy) * 1.0 / simsum

m = max(len(rs), len(ss))
for i in range(m):
	temp = 0
	if i < len(rs):
		temp += rn[i]
	if i < len(ss):
		temp -= sn[i]
	logmae[3] += np.log(abs(temp))
	square[1] += np.log(abs(temp) * 1.1 ** i)

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_num_bin.png')
plt.cla()

fw = open(prefix+'BigSimCascades', 'w')
for line in bigsim:
	fw.write(line)
fw.close()

fw = open(prefix+'BigRealCascades', 'w')
for line in bigreal:
	fw.write(line)
fw.close()

print logmae
print square
