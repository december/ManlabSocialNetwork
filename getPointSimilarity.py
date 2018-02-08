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

simulation = False
filename = 'Real'
if len(sys.argv) > 1:
	filename = sys.argv[1]
	simulation = True

def calcPJ(x, y, cnt):
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
fr = open(relation_prefix+'relations'+suffix, 'r')
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
			#isroot = int(root_tweet.has_key(data[3]))
			isroot = 0
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

fw1 = open(prefix+'similarity/'+filename+'_pearson_point.detail', 'w')
fw2 = open(prefix+'similarity/'+filename+'_jaccard_point.detail', 'w')

for k in relation_dic:
	tempcnt = len(relation_dic[k])
	if tempcnt < 5:
		continue
	valueset = [-1, -1, -1, -1]
	m0 = 0
	m1 = 0
	if realdic[0].has_key(k):
		m0 = len(realdic[0][k])
	if realdic[1].has_key(k):
		m1 = len(realdic[1][k])
	if m0 > 1:
		pearson = 0.0
		jaccard = 0.0
		pointlist = realdic[0][k].keys()
		number = m0 * (m0 - 1) / 2
		for i in range(m0):
			for j in range(i+1, m0):
				if len(realdic[0][k][pointlist[i]]) == tempcnt or len(realdic[0][k][pointlist[j]]) == tempcnt:
					number -= 1
					continue
				pij, jij = calcPJ(realdic[0][k][pointlist[i]], realdic[0][k][pointlist[j]], tempcnt)
				pearson += abs(pij)
				jaccard += jij
		if number > 0:
			valueset[0] = pearson / number
			valueset[2] = jaccard / number
	if m1 > 1:
		pearson = 0.0
		jaccard = 0.0
		pointlist = realdic[1][k].keys()
		number = m1 * (m1 - 1) / 2
		for i in range(m1):
			for j in range(i+1, m1):
				if len(realdic[1][k][pointlist[i]]) == tempcnt or len(realdic[1][k][pointlist[j]]) == tempcnt:
					number -= 1
					continue
				pij, jij = calcPJ(realdic[1][k][pointlist[i]], realdic[1][k][pointlist[j]], tempcnt)
				pearson += abs(pij)
				jaccard += jij
		if number > 0:
			valueset[1] = pearson / number
			valueset[3] = jaccard / number
	fw1.write(k+'\t'+str(valueset[0])+'\t'+str(valueset[1])+'\n')
	fw2.write(k+'\t'+str(valueset[2])+'\t'+str(valueset[3])+'\n')
fw1.close()
fw2.close()

