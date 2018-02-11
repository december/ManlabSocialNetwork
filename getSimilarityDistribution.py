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

plt.yscale('log')
plt.style.use("ggplot")
plt.plot(px, py, 'r')
plt.plot(px, ps, 'b')
plt.xlabel(u'Pearson')
plt.ylabel(u'Distribution')
plt.savefig(prefix+'similarity/'+filename+'_pearson_num.png')
plt.cla()

plt.yscale('log')
plt.style.use("ggplot")
plt.plot(jx, jy, 'r')
plt.plot(jx, js, 'b')
plt.xlabel(u'Jaccard')
plt.ylabel(u'Distribution')
plt.savefig(prefix+'similarity/'+filename+'_jaccard_num.png')
plt.cla()

px = np.array(ppos)
py = np.array(pdb_cum) * 1.0 / psum
ps = np.array(pdb_sim_cum) * 1.0 / psum_sim
jx = np.array(jpos)
jy = np.array(jdb_cum) * 1.0 / jsum
js = np.array(jdb_sim_cum) * 1.0 / jsum_sim

plt.yscale('log')
plt.style.use("ggplot")
plt.plot(px, py, 'r')
plt.plot(px, ps, 'b')
plt.xlabel(u'Pearson')
plt.ylabel(u'Distribution')
plt.savefig(prefix+'similarity/'+filename+'_pearson_cum.png')
plt.cla()

plt.yscale('log')
plt.style.use("ggplot")
plt.plot(jx, jy, 'r')
plt.plot(jx, js, 'b')
plt.xlabel(u'Jaccard')
plt.ylabel(u'Distribution')
plt.savefig(prefix+'similarity/'+filename+'_jaccard_cum.png')
plt.cla()
