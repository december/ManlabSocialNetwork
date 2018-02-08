import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
import os

root_tweet = {}
realdic = list()
realdic.append({}) #retweet
realdic.append({}) #post root

authordic = {} #from tweet id to author id
cnt = 0 #cascade number

simulation = False
filename = 'Real'
if len(sys.argv) > 1:
	filename = sys.argv[1]
	simulation = True

def calcPJ(x, y):
	p = 0
	j = 0
	xlist = x.keys()
	ylist = y.keys()
	xset = set(xlist)
	yset = set(ylist)
	inter = list(xset.intersection(yset))
	if not len(inter) == 0:
		union = list(xset.union(yset))
		j = len(inter) * 1.0 / len(union)
	#top = 0
	#for key in inter:
	#	top += x[key] * y[key]
	#top *= cnt
	#xsum = sum(x.values())
	#ysum = sum(y.values())
	xlen = len(xlist)
	ylen = len(ylist)
	top = len(inter) * cnt
	top -= xlen * ylen
	bottom = math.sqrt(cnt * xlen - xlen ** 2) * math.sqrt(cnt * ylen - ylen ** 2)
	p = top * 1.0 / bottom
	return p, j

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268
realdata = list()
relation_prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'

relation_dic = {}
fr = open(relation_prefix+'relations.detal', 'r')
relationdata = fr.readlines()
n = len(relationdata)
i = 0
while i < n:
	temp = relationdata[i].split('\t')
	number = int(temp[1]) + 1
	if not relation_dic.has_key(temp[0]):
		relation_dic[temp[0]] = {}
	fdidx = 0
	for j in range(i+1, i+number):
		data = relationdata[j].split('\t')
		relation_dic[temp[0]][data[1]] = fdidx
		fdidx += 1
	i += number
fr.close()


if not simulation:
	namelist = os.listdir(path)
	for name in namelist:
		fr = open(path+name, 'r')
		realdata.extend(fr.readlines())
		fr.close()
else:
	fr = open(prefix+'result/'+filename+'.detail', 'r')
	realdata = fr.readlines()
	fr.close()

n = len(realdata)
i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	for j in range(i+1, i+number):
		data = realdata[j][:-1].split('\t')
		if data[3] != '-1':
			isroot = int(root_tweet.has_key(data[3]))
			if not realdic[isroot].has_key(data[4]):
				realdic[isroot][data[4]] = {}
			if not realdic[isroot][data[4]].has_key(data[3]):
				realdic[isroot][data[4]][data[3]] = {}
			realdic[isroot][data[4]][data[3]][relation_dic[data[4]][data[1]]] = 1
		else:
			root_tweet[data[0]] = 1
	i += number
m1 = len(realdic[0])
m2 = len(realdic[1])
print 'Construct vectors finished.'
print [m1, m2]

#pearson = np.zeros((m, m))
#jaccard = np.zeros((m, m))
pearson = 0
jaccard = 0
numbers = m * (m - 1) / 2
pointlist = realdic.keys()
bins = 10000
pdb = np.zeros(bins)
jdb = np.zeros(bins)
ppos = np.zeros(bins)
jpos = np.zeros(bins)
wid1 = 1.0 / bins
wid2 = 2.0 / bins
end1 = wid1
end2 = -1 + wid2
for i in range(bins):
	ppos[i] = end2
	jpos[i] = end1
	end1 += wid1
	end2 += wid2

fw1 = open(prefix+'similarity/'+filename+'_pearson_point.detail', 'w')
fw2 = open(prefix+'similarity/'+filename+'_jaccard_point.detail', 'w')

for k in relation_dic:
	m0 = len(relation_dic[0][k])
	m1 = len(relation_dic[1][k])
	valueset = [-1, -1, -1, -1]
	if m0 > 1:
		pearson = 0
		jaccard = 0
		pointlist = relation_dic[0][k].keys()
		for i in range(m0):
			for j in range(i+1, m0):
				pij, jij = calcPJ(relation_dic[0][k][pointlist[i]], relation_dic[0][k][pointlist[i]])
				pearson += pij
				jaccard += jij
		valueset[0] = pearson * 2.0 / m0 / (m0 - 1)
		valueset[2] = jaccard * 2.0 / m0 / (m0 - 1)
	if m1 > 1:
		pearson = 0
		jaccard = 0
		pointlist = relation_dic[1][k].keys()
		for i in range(m1):
			for j in range(i+1, m1):
				pij, jij = calcPJ(relation_dic[1][k][pointlist[i]], relation_dic[1][k][pointlist[i]])
				pearson += pij
				jaccard += jij
		valueset[1] = pearson * 2.0 / m1 / (m1 - 1)
		valueset[3] = jaccard * 2.0 / m1 / (m1 - 1)
	fw1.write(relation_dic[k]+'\t'+valueset[0]+'\t'+valueset[2]+'\n')
	fw2.write(relation_dic[k]+'\t'+valueset[1]+'\t'+valueset[3]+'\n')
fw1.close()
fw2.close()

