import scipy as sp
import numpy as np

users = 7268
allusers = 7268
edges = 7268
edgemap = {}
uid = list()
iddic = {}

lbd = np.zeros(users) #parameter lambda which have calculated before
omega = np.zeros(allusers) #parameter omega
phi1 = np.zeros(allusers) #one of topic distribution
phi2 = np.zeros(allusers) #one of topic distribution
phi3 = np.zeros(allusers) #one of topic distribution
phi4 = np.zeros(allusers) #one of topic distribution
phi5 = np.zeros(allusers) #one of topic distribution
pi = np.zeros(edges) #parameter pi (based on edges), row is sender while col is receiver
x = np.zeros(edges) #parameter x (based on edges), row is sender while col is receiver


prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'

fr = open(prefix+'lambda_Poisson'+suffix, 'r')
lbdlist = fr.readlines()
for i in range(users):
	temp = lbdlist[i].split('\t')
	uid.append(temp[0])
	iddic[temp[0]] = i
	lbd[i] = float(temp[1])
fr.close()

fr = open(prefix+'omega_Poisson'+suffix, 'r')
omglist = fr.readlines()
for i in range(allusers):
	temp = lbdlist[i].split('\t')
	omega[i] = float(temp[1])
fr.close()

for i in range(5):
	fr = open(prefix+'phi'+str(i)+'_Poisson'+suffix, 'r')
	philist = fr.readlines()
	for j in range(allusers):
		temp = philist[j].split('\t')
		if i == 0:
			phi1[j] = float(temp[1])
		if i == 1:
			phi2[j] = float(temp[1])
		if i == 2:
			phi3[j] = float(temp[1])
		if i == 3:
			phi4[j] = float(temp[1])
		if i == 4:
			phi5[j] = float(temp[1])
	fr.close()

fr = open(prefix+'pi_Poisson'+suffix, 'r')
pilist = fr.readlines()
for i in range(edges):
	temp = lbdlist[i].split('\t')
	row = iddic[int(temp[0])]
	col = iddic[int(temp[1])]
	if not edgemap.has_key(row):
		edgemap[row] = {}
	edgemap[row][col] = i
	pi[i] = float(temp[2])
fr.close()

fr = open(prefix+'x_Poisson'+suffix, 'r')
xlist = fr.readlines()
for i in range(edges):
	temp = lbdlist[i].split('\t')
	x[i] = float(temp[2])
fr.close()
