import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

real = {}
fr = open('../../cascading_generation_model/all/all.detail', 'r')
data = fr.readlines()
fr.close()
n = len(data)
i = 0
while i < n:
	temp = data[i].split('\t')
	number = int(temp[1])
	if real.has_key(number):
		real[number] += 1
	else:
		real[number] = 1
	i += number + 1

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
plt.savefig('../../cascading_generation_model/all/All.png')
plt.cla()

plt.xscale('log')
plt.yscale('log')
plt.plot(rs, rn, 'ro', label='Real')
plt.xlabel(u'Size')
plt.ylabel(u'Distribution')
plt.legend(loc='upper right');  
plt.savefig('../../cascading_generation_model/all/All_log.png')
plt.cla()