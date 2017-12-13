import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268

namelist = os.listdir(path)
real = {}
sim = {}
for name in namelist:
	fr = open(path+name, 'r')
	realdata = fr.readlines()
	n = len(realdata)
	i = 0
	while i < n:
		temp = realdata[i].split('\t')
		number = int(temp[1])
		if real.has_key(number):
			real[number] += 1
		else:
			real[number] = 1
		i += number + 1
	fr.close()

realsize = sorted(real.keys())
realnum = list()
for size in realsize:
	realnum.append(real[size])

realsum = sum(realnum)

rs = np.array(realsize)
rn = np.array(realnum) * 1.0 / realsum

plt.plot(rs, rn, 'ro', label='Real')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
plt.savefig(prefix+'RealSize/All.png')
plt.cla()

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
plt.savefig(prefix+'RealSize/All_log.png')
plt.cla()