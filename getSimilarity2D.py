import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
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
m = 7625

p_matrix_real = np.zeros((m, m))
j_matrix_real = np.zeros((m, m))
p_matrix_sim = np.zeros((m, m))
j_matrix_sim = np.zeros((m, m))

fr1 = open(prefix+'similarity/'+filename+'_pearson_value.detail', 'r')
data = fr1.readlines()
n = len(data)
for i in range(n):
	temp = data[i].split('\t')[:-1]
	for j in range(n):
		p_matrix_real[i][j] = float(temp[j]) 
fr1.close()

fr2 = open(prefix+'similarity/'+filename+'_jaccard_value.detail', 'r')
data = fr2.readlines()
n = len(data)
for i in range(n):
	temp = data[i].split('\t')[:-1]
	for j in range(n):
		j_matrix_real[i][j] = float(temp[j]) 
fr2.close()

filename = sys.argv[1]
fr1 = open(prefix+'similarity/'+filename+'_pearson_value.detail', 'r')
data = fr1.readlines()
n = len(data)
for i in range(n):
	temp = data[i].split('\t')[:-1]
	for j in range(n):
		p_matrix_sim[i][j] = float(temp[j]) 
fr1.close()

fr2 = open(prefix+'similarity/'+filename+'_jaccard_value.detail', 'r')
data = fr2.readlines()
n = len(data)
for i in range(n):
	temp = data[i].split('\t')[:-1]
	for j in range(n):
		j_matrix_sim[i][j] = float(temp[j]) 
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

delta_p = np.zeros((n, n))
delta_j = np.zeros((n, n))
for i in range(n):
	for j in range(n):
		delta_p[i][j] = abs(p_matrix_real[i][j] - p_matrix_sim[i][j])
		delta_j[i][j] = abs(j_matrix_real[i][j] - j_matrix_sim[i][j])

sns.set()
ax = sns.heatmap(delta_p[:1000][:1000], norm=LogNorm(vmin=delta_p.min(), vmax=delta_p.max()))
plt.savefig(prefix+'similarity/'+filename+'_pearson_2D.png')
plt.cla()
#df = pd.DataFrame(j_matrix)
#sns.heatmap(df, vmin=np.min(p_matrix), vmax=np.max(p_matrix))
#plt.savefig(prefix+'similarity/'+filename+'_jaccard_2D.png')
#plt.cla()

