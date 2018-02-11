import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
import os

realdic = {} #from id to its cascade dic
relation = {} #from id to follower id
authordic = {} #from tweet id to author id
cnt = 0 #cascade number
mode = 0

simulation = False
filename = 'Real'

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268
realdata = list()
relation_prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'

bins = 50
pdb = np.zeros(bins)
pdb_cum = np.zeros(bins)
pdb_sim = np.zeros(bins)
pdb_sim_cum = np.zeros(bins)
pdb_sim1 = np.zeros(bins)
pdb_sim1_cum = np.zeros(bins)
pdb_sim2 = np.zeros(bins)
pdb_sim2_cum = np.zeros(bins)


jdb = np.zeros(bins)
jdb_cum = np.zeros(bins)
jdb_sim = np.zeros(bins)
jdb_sim_cum = np.zeros(bins)
ppos = np.zeros(bins)
jpos = np.zeros(bins)
wid = 1.0 / bins
end = wid / 2
for i in range(bins):
	ppos[i] = end
	jpos[i] = end
	end += wid

fr1 = open(prefix+'similarity/'+filename+'_pearson_point.detail', 'r')
data = fr1.readlines()
for line in data:
	temp = line.split('\t')
	if mode < 2:
		p = float(temp[mode+1])
	else:
		p = abs(float(temp[1])-float(temp[2]))
		if float(temp[1]) == -1 or float(temp[2]) == -1:
			p = -1
	if p < 0:
		continue
	idx = int(p / wid)
	pdb[min(idx, bins-1)] += 1
fr1.close()

psum = sum(pdb)
temps = psum
pdb_cum[0] = temps
for i in range(1, bins):
	temps -= pdb[i-1]
	pdb_cum[i] = temps


filename = 'All_parameter_500'
fr1 = open(prefix+'similarity/'+filename+'_pearson_point.detail', 'r')
data = fr1.readlines()
for line in data:
	temp = line.split('\t')
	if mode < 2:
		p = float(temp[mode+1])
	else:
		p = abs(float(temp[1])-float(temp[2]))
		if float(temp[1]) == -1 or float(temp[2]) == -1:
			p = -1
	if p < 0:
		continue
	idx = int(p / wid)
	pdb_sim[min(idx, bins-1)] += 1
fr1.close()

psum_sim = sum(pdb_sim)
temps = psum_sim
pdb_sim_cum[0] = temps
for i in range(1, bins):
	temps -= pdb_sim[i-1]
	pdb_sim_cum[i] = temps

filename = 'BranchingProcess'
fr1 = open(prefix+'similarity/'+filename+'_pearson_point.detail', 'r')
data = fr1.readlines()
for line in data:
	temp = line.split('\t')
	if mode < 2:
		p = float(temp[mode+1])
	else:
		p = abs(float(temp[1])-float(temp[2]))
		if float(temp[1]) == -1 or float(temp[2]) == -1:
			p = -1
	if p < 0:
		continue
	idx = int(p / wid)
	pdb_sim1[min(idx, bins-1)] += 1
fr1.close()

psum_sim1 = sum(pdb_sim1)
temps = psum_sim1
pdb_sim1_cum[0] = temps
for i in range(1, bins):
	temps -= pdb_sim1[i-1]
	pdb_sim1_cum[i] = temps

filename = 'NoTopic'
fr1 = open(prefix+'similarity/'+filename+'_pearson_point.detail', 'r')
data = fr1.readlines()
for line in data:
	temp = line.split('\t')
	if mode < 2:
		p = float(temp[mode+1])
	else:
		p = abs(float(temp[1])-float(temp[2]))
		if float(temp[1]) == -1 or float(temp[2]) == -1:
			p = -1
	if p < 0:
		continue
	idx = int(p / wid)
	pdb_sim2[min(idx, bins-1)] += 1
fr1.close()

psum_sim2 = sum(pdb_sim2)
temps = psum_sim2
pdb_sim2_cum[0] = temps
for i in range(1, bins):
	temps -= pdb_sim2[i-1]
	pdb_sim2_cum[i] = temps

px = np.array(ppos)
py = np.array(pdb_cum) * 1.0 / psum
ps = np.array(pdb_sim_cum) * 1.0 / psum_sim
ps1 = np.array(pdb_sim1_cum) * 1.0 / psum_sim1
ps2 = np.array(pdb_sim2_cum) * 1.0 / psum_sim2

#plt.yscale('log')
plt.plot(px, py, '#ff6347', label='Real', linewidth=2.5)
plt.plot(px, ps, '#4876ff', label='Our Framework', linewidth=2.5)
plt.plot(px, ps1, '#8c8c8c',linestyle='--',label='BP', linewidth=2.5)
plt.plot(px, ps2, '#458b00',linestyle='--',label='Base', linewidth=2.5)
plt.plot([0.2, 0.2,], [0, py[9]], 'k:', linewidth=2.5)
plt.ylim((0,1))
plt.xlabel(u'Collectivity', fontsize=14)
plt.ylabel(u'CDF', fontsize=14)
plt.legend(loc='upper right', fontsize=15);
plt.savefig(prefix+'similarity/'+str(mode)+'_'+filename+'_collectivity.png')
plt.cla()
