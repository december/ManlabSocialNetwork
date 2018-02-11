import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
sns.set()

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
#print realsize
realnum = list()
realcum = [1]
for size in realsize:
	realnum.append(real[size])
n = len(realnum)
s = sum(realnum)
for i in range(n-1):
	s -= realnum[i]
	realcum.append(s)

simsize = sorted(sim.keys())
#print simsize
simnum = list()
simcum = [1]
for size in simsize:
	simnum.append(sim[size] * 1.0 / m)
n = len(simnum)
s = sum(simnum)
for i in range(n-1):
	s -= simnum[i]
	simcum.append(s)

realsum = sum(realnum)
simsum = sum(simnum)
#print simcum
#print realcum

print realsize
newcum = list()
newnum = list()
num = len(realsize)
for i in range(num):
	newcum.append(realcum[i] * 1.0 / realsum)
	newnum.append(realnum[i] * 1.0 / realsum)
print newcum
print newnum	 

print simsize
newcum = list()
newnum = list()
num = len(simsize)
for i in range(num):
	newcum.append(simcum[i] * 1.0 / simsum)
	newnum.append(simnum[i] * 1.0 / simsum)
print newcum
print newnum	 



logmae = [0,0]
rs = np.array(realsize)
rn = np.array(realcum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simcum) * 1.0 / simsum

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
		cr = 1 - rn[pos1]
	if pos2 < len(ss) and ss[pos2] == i:
		cs = 1 - sn[pos2]
	if pos1 >= len(rs):
		cr = 1
	if pos2 >= len(ss):
		cs = 1
	logmae[0] += abs(np.log(cr) - np.log(cs))
logmae[0] = logmae[0] / m

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_cum.png', dpi=600)
plt.cla()


rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
'''
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
		temp += np.log(rn[pos1]+1)
	if pos2 < len(ss) and ss[pos2] == i:
		temp -= np.log(sn[pos2]+1)
	if temp != 0:
		logmae[1] += abs(temp)
'''
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_num.png', dpi=600)
plt.cla()

square = [0, 0]
binrx, binry = GetBin(1.1, realsize, realcum) 
binsx, binsy = GetBin(1.1, simsize, simcum)
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
		cr = 1 - rn[i]
	if i < len(ss):
		cs = 1 - sn[i]
	logmae[1] += abs(np.log(cr) - np.log(cs))
	square += abs(np.log(cr) - np.log(cs)) * 1.1 ** (i+1)
logmae[1] = logmae[1] / m
square = square / m

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_cum_bin.png', dpi=600)
plt.cla()

binrx, binry = GetBin(1.1, realsize, realnum) 
binsx, binsy = GetBin(1.1, simsize, simnum)
rs = np.array(binrx)
rn = np.array(binry) * 1.0 / realsum
ss = np.array(binsx)
sn = np.array(binsy) * 1.0 / simsum
'''
m = max(len(rs), len(ss))
for i in range(m):
	temp = 0
	if i < len(rs):
		temp += np.log(rn[i]+1)
	if i < len(ss):
		temp -= np.log(sn[i]+1)
	if temp != 0:
		logmae[3] += abs(temp)
		square[1] += abs(temp) * 1.1 ** (i+1)
'''
plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, 'bo', label='Sim')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
if not single:
	filename = 'all'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_num_bin.png', dpi=600)
plt.cla()

fw = open(prefix+'BigSimCascades', 'w')
for line in bigsim:
	fw.write(line)
fw.close()

fw = open(prefix+'BigRealCascades', 'w')
for line in bigreal:
	fw.write(line)
fw.close()

#print logmae
#print square
