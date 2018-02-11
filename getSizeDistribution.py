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


fr = open('/home/luyunfei/cascading_generation_model/simulation/result/All_parameter_500.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
i = 0
while i < n:
	temp = simdata[i].split('\t')
	number = int(temp[1])
	a = int(simdata[i+1].split('\t')[1])
	if sim.has_key(number):
		sim[number] += 1
	else:
		sim[number] = 1
	i += number + 1
fr.close()

sim1 = {}
fr = open('/home/luyunfei/cascading_generation_model/simulation/result/BranchingProcess.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
i = 0
while i < n:
	temp = simdata[i].split('\t')
	number = int(temp[1])
	a = int(simdata[i+1].split('\t')[1])
	if sim1.has_key(number):
		sim1[number] += 1
	else:
		sim1[number] = 1
	i += number + 1
fr.close()

sim2 = {}
fr = open('/home/luyunfei/cascading_generation_model/simulation/result/EpidemicModel.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
i = 0
while i < n:
	temp = simdata[i].split('\t')
	number = int(temp[1])
	a = int(simdata[i+1].split('\t')[1])
	if sim2.has_key(number):
		sim2[number] += 1
	else:
		sim2[number] = 1
	i += number + 1
fr.close()

sim3 = {}
fr = open('/home/luyunfei/cascading_generation_model/simulation/result/NoTopic.detail', 'r')
simdata = fr.readlines()
n = len(simdata)
i = 0
while i < n:
	temp = simdata[i].split('\t')
	number = int(temp[1])
	a = int(simdata[i+1].split('\t')[1])
	if sim3.has_key(number):
		sim3[number] += 1
	else:
		sim3[number] = 1
	i += number + 1
fr.close()



realsize = sorted(real.keys())
#print realsize
realnum = list()
realcum = [sum(realnum)]
for size in realsize:
	realnum.append(real[size])
n = len(realnum)
s = sum(realnum)
for i in range(n-1):
	s -= realnum[i]
	realcum.append(s)

m=1
simsize = sorted(sim.keys())
#print simsize
simnum = list()
simcum = [sum(simnum)]
for size in simsize:
	simnum.append(sim[size] * 1.0 / m)
n = len(simnum)
s = sum(simnum)
for i in range(n-1):
	s -= simnum[i]
	simcum.append(s)

simsize1 = sorted(sim1.keys())
#print simsize
simnum1 = list()
simcum1 = [sum(simnum1)]
for size in simsize1:
	simnum1.append(sim1[size] * 1.0 / m)
n = len(simnum1)
s = sum(simnum1)
for i in range(n-1):
	s -= simnum1[i]
	simcum1.append(s)

simsize2 = sorted(sim2.keys())
#print simsize
simnum2 = list()
simcum2 = [sum(simnum1)]
for size in simsize2:
	simnum2.append(sim1[size] * 1.0 / m)
n = len(simnum2)
s = sum(simnum2)
for i in range(n-1):
	s -= simnum2[i]
	simcum2.append(s)

simsize3 = sorted(sim3.keys())
#print simsize
simnum3 = list()
simcum3 = [sum(simnum3)]
for size in simsize3:
	simnum3.append(sim3[size] * 1.0 / m)
n = len(simnum3)
s = sum(simnum3)
for i in range(n-1):
	s -= simnum3[i]
	simcum3.append(s)

realsum = sum(realnum)
simsum = sum(simnum)
simsum1 = sum(simnum1)
simsum2 = sum(simnum2)
simsum3 = sum(simnum3)
#print simcum
#print realcum


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

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.plot(ss, sn, '#4876ff', label='Our framework')
plt.plot(ss1, sn1, '#8c8c8c', label='BP')
plt.plot(ss2, sn2, '#ffa500', label='EP')
plt.plot(ss3, sn3, '#458b00', label='Base')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
filename = 'all_size'
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
plt.plot(ss1, sn1, '#8c8c8c', label='BP')
plt.plot(ss2, sn2, '#ffa500', label='EP')
plt.plot(ss3, sn3, '#458b00', label='Base')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
filename = 'all_size'
plt.savefig(prefix+'SizeDistribution/'+str(filename)+'_cum.png', dpi=600)
plt.cla()

