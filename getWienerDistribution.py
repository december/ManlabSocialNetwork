import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns

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


fr = open('/home/luyunfei/cascading_generation_model/simulation/result/All_parameter_500.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
sim = {}
i = 0
num = 1
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

fr = open('/home/luyunfei/cascading_generation_model/simulation/result/BranchingProcess.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
sim1 = {}
i = 0
num = 1
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
	if sim1.has_key(wi):
		sim1[wi] += 1
	else:
		sim1[wi] = 1
	i += number
fr.close()

fr = open('/home/luyunfei/cascading_generation_model/simulation/result/EpidemicModel.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
sim2 = {}
i = 0
num = 1
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
	if sim2.has_key(wi):
		sim2[wi] += 1
	else:
		sim2[wi] = 1
	i += number
fr.close()

fr = open('/home/luyunfei/cascading_generation_model/simulation/result/NoTopic.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
sim3 = {}
i = 0
num = 1
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
	if sim3.has_key(wi):
		sim3[wi] += 1
	else:
		sim3[wi] = 1
	i += number
fr.close()

realsize = sorted(real.keys())
realnum = list()
for size in realsize:
	realnum.append(real[size])

simsize = sorted(sim.keys())
simnum = list()
for size in simsize:
	simnum.append(sim[size] * 1.0)

simsize1 = sorted(sim1.keys())
simnum1 = list()
for size in simsize1:
	simnum1.append(sim1[size] * 1.0)

simsize2 = sorted(sim2.keys())
simnum2 = list()
for size in simsize2:
	simnum2.append(sim2[size] * 1.0)

simsize3 = sorted(sim3.keys())
simnum3 = list()
for size in simsize3:
	simnum3.append(sim3[size] * 1.0)

realsize, realnum = GetBin(1.4, realsize, realnum) 
simsize, simnum = GetBin(1.4, simsize, simnum)
simsize1, simnum1 = GetBin(1.4, simsize1, simnum1)
simsize2, simnum2 = GetBin(1.4, simsize2, simnum2)
simsize3, simnum3 = GetBin(1.4, simsize3, simnum3)

realsum = sum(realnum)
simsum = sum(simnum)
simsum1 = sum(simnum1)
simsum2 = sum(simnum2)
simsum3 = sum(simnum3)


realcum = [sum(realnum)]
n = len(realnum)
s = sum(realnum)
for i in range(n-1):
	s -= realnum[i]
	realcum.append(s)

simcum = [sum(simnum)]
n = len(simnum)
s = sum(simnum)
for i in range(n-1):
	s -= simnum[i]
	simcum.append(s)

simcum1 = [sum(simnum1)]
n = len(simnum1)
s = sum(simnum1)
for i in range(n-1):
	s -= simnum1[i]
	simcum1.append(s)

simcum2 = [sum(simnum2)]
n = len(simnum2)
s = sum(simnum2)
for i in range(n-1):
	s -= simnum2[i]
	simcum2.append(s)

simcum3 = [sum(simnum3)]
n = len(simnum3)
s = sum(simnum3)
for i in range(n-1):
	s -= simnum3[i]
	simcum3.append(s)

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simnum) * 1.0 / simsum
ss1 = np.array(simsize1)
sn1 = np.array(simnum1) * 1.0 / simsum1
ss2 = np.array(simsize2)
sn2 = np.array(simnum2) * 1.0 / simsum2
ss3 = np.array(simsize3)
sn3 = np.array(simnum3) * 1.0 / simsum3

sns.set()
sns.set_style('white')

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, '#4876ff', label='Our framework')
plt.plot(ss1, sn1, '#8c8c8c', linestyle='--', label='BP')
plt.plot(ss2, sn2, '#ffa500', label='EP')
plt.plot(ss3, sn3, '#458b00', linestyle='--', label='Base')
plt.xlabel(u'Wiener Index', fontsize=14)
plt.ylabel(u'PDF', fontsize=14)
plt.legend(loc='upper right', fontsize=15);  
filename = 'all_wiener'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_num.png', dpi=600)
plt.cla()

#print logmae
#print square
rs = np.array(realsize)
rn = np.array(realcum) * 1.0 / realsum
ss = np.array(simsize)
sn = np.array(simcum) * 1.0 / simsum
ss1 = np.array(simsize1)
sn1 = np.array(simcum1) * 1.0 / simsum1
ss2 = np.array(simsize2)
sn2 = np.array(simcum2) * 1.0 / simsum2
ss3 = np.array(simsize3)
sn3 = np.array(simcum3) * 1.0 / simsum3

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, '#4876ff', label='Our framework')
plt.plot(ss1, sn1, '#8c8c8c', linestyle='--', label='BP')
plt.plot(ss2, sn2, '#ffa500', label='EP')
plt.plot(ss3, sn3, '#458b00', linestyle='--', label='Base')
plt.xlabel(u'Wiener Index', fontsize=14)
plt.ylabel(u'CDF', fontsize=14)
plt.legend(loc='upper right', fontsize=15);  
filename = 'all_wiener'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_cum.png', dpi=600)
plt.cla()

'''
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
'''
