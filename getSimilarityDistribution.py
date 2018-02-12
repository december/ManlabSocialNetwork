import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
import os
import seaborn as sns
sns.set()
sns.set_style('white')

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

realdic = {} #from id to its cascade dic
relation = {} #from id to follower id
authordic = {} #from tweet id to author id
cnt = 0 #cascade number

simulation = False
filename = 'Real'

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268
realdata = list()
relation_prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'

bins = 100
pdb = np.zeros(bins)
pdb_cum = np.zeros(bins)
pdb_sim = np.zeros(bins)
pdb_sim_cum = np.zeros(bins)
jdb = np.zeros(bins)
jdb_cum = np.zeros(bins)
jdb_sim = np.zeros(bins)
jdb_sim_cum = np.zeros(bins)
ppos = np.zeros(bins)
jpos = np.zeros(bins)
wid = 0.01
end = 0.005
for i in range(bins):
	ppos[i] = end
	jpos[i] = end
	end += wid

fr1 = open(prefix+'similarity/'+filename+'_pearson.detail', 'r')
data = fr1.readlines()
for line in data:
	temp = line.split('\t')
	p = float(temp[2])
	if p < 0:
		continue
	idx = int(p / wid)
	pdb[min(idx, 99)] += 1
fr1.close()

fr2 = open(prefix+'similarity/'+filename+'_jaccard.detail', 'r')
data = fr2.readlines()
for line in data:
	temp = line.split('\t')
	p = float(temp[2])
	idx = int(p / wid)
	jdb[min(idx, 99)] += 1
fr2.close()

psum = sum(pdb)
jsum = sum(jdb)
temps = psum
pdb_cum[0] = temps
for i in range(1, bins):
	temps -= pdb[i-1]
	pdb_cum[i] = temps
temps = jsum
jdb_cum[0] = temps
for i in range(1, bins):
	temps -= jdb[i-1]
	jdb_cum[i] = temps

filename = sys.argv[1]
fr1 = open(prefix+'similarity/'+filename+'_pearson.detail', 'r')
data = fr1.readlines()
for line in data:
	temp = line.split('\t')
	p = float(temp[2])
	if p < 0:
		continue
	idx = int(p / wid)
	pdb_sim[min(idx, 99)] += 1
fr1.close()

fr2 = open(prefix+'similarity/'+filename+'_jaccard.detail', 'r')
data = fr2.readlines()
for line in data:
	temp = line.split('\t')
	p = float(temp[2])
	idx = int(p / wid)
	jdb_sim[min(idx, 99)] += 1
fr2.close()

psum_sim = sum(pdb_sim)
jsum_sim = sum(jdb_sim)
temps = psum_sim
pdb_sim_cum[0] = temps
for i in range(1, bins):
	temps -= pdb_sim[i-1]
	pdb_sim_cum[i] = temps
temps = jsum_sim
jdb_sim_cum[0] = temps
for i in range(1, bins):
	temps -= jdb_sim[i-1]
	jdb_sim_cum[i] = temps	

px = np.array(ppos)
py = np.array(pdb) * 1.0 / psum
ps = np.array(pdb_sim) * 1.0 / psum_sim
jx = np.array(jpos)
jy = np.array(jdb) * 1.0 / jsum
js = np.array(jdb_sim) * 1.0 / jsum_sim

#px1, py = GetBin(0.1, px, py) 
#px2, ps = GetBin(0.1, px, ps) 
#jx1, jy = GetBin(0.1, jx, jy) 
#jx2, js = GetBin(0.1, jx, js) 

#plt.xscale('log')
plt.yscale('log')
plt.style.use("ggplot")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.plot(px, py, '#ff6347', linestyle='+', label='Real', linewidth=2.5)
plt.plot(px, py, 'ro', label='Real')
plt.plot(px, ps, '#4876ff', label='Our Method', linewidth=2.5)
plt.xlabel(u'Pearson Coeffecient', fontsize=14)
plt.ylabel(u'PDF', fontsize=14)
plt.legend(loc='upper right', fontsize=20);
plt.tight_layout()
#plt.title('Distribution of Pearson Coeffecient', fontsize=20)
plt.savefig(prefix+'similarity/'+filename+'_pearson_num.eps', dpi=1200)
plt.cla()

#plt.xscale('log')
plt.yscale('log')
plt.style.use("ggplot")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.plot(jx, jy, '#ff6347', linestyle='+', label='Real', linewidth=2.5)
plt.plot(jx, jy, 'ro', label='Real')
plt.plot(jx, js, '#4876ff', label='Our Method', linewidth=2.5)
plt.xlabel(u'Jaccard Index', fontsize=14)
plt.ylabel(u'PDF', fontsize=14)
plt.legend(loc='upper right', fontsize=20);
plt.tight_layout()
plt.savefig(prefix+'similarity/'+filename+'_jaccard_num.eps', dpi=1200)
plt.cla()

'''
print py[0]
print ps[0]
print jy[0]
print js[0]
'''

'''

px = np.array(ppos)
py = np.array(pdb_cum) * 1.0 / psum
ps = np.array(pdb_sim_cum) * 1.0 / psum_sim
jx = np.array(jpos)
jy = np.array(jdb_cum) * 1.0 / jsum
js = np.array(jdb_sim_cum) * 1.0 / jsum_sim

plt.yscale('log')
plt.style.use("ggplot")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(px, py, '#ff6347', label='Real', linewidth=2)
plt.plot(px, ps, '#4876ff', label='Base', linewidth=2)
plt.xlabel(u'Pearson Coeffecient', fontsize=14)
plt.ylabel(u'PDF', fontsize=14)
plt.legend(loc='upper right', fontsize=20);
plt.savefig(prefix+'similarity/'+filename+'_pearson_cum.eps', dpi=1200)
plt.cla()

plt.yscale('log')
plt.style.use("ggplot")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(jx, jy, '#ff6347', label='Real', linewidth=2)
plt.plot(jx, js, '#4876ff', label='Base', linewidth=2)
plt.xlabel(u'Jaccard Index', fontsize=14)
plt.ylabel(u'PDF', fontsize=14)
plt.legend(loc='upper right', fontsize=20);
plt.savefig(prefix+'similarity/'+filename+'_jaccard_cum.eps', dpi=1200)
plt.cla()
'''