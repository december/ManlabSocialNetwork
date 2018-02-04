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


def randomSelect(person):
	friend = random.choice(relation[person])
	return calcPearson(person, friend)

def chooseTwo(person):
	friend = relation[person]
	#f1 = random.choice(friend)
	#friend.remove(f1)
	#f2 = random.choice(friend)
	#friend.append(f1)
	f = random.sample(friend, 2)
	return calcPearson(f[0], f[1])

filename = int(sys.argv[1])
single = False

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268
relation_prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
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
		data = realdata[j].split('\t')
		if not realdic.has_key(data[1]):
			realdic[data[1]] = {}
		realdic[data[1]][cnt] = 1
		#if not realdic[data[1]].has_key(cnt)
		#	realdic[data[1]][cnt] = 1
		#else:
		#	realdic[data[1]][cnt] += 1
	cnt += 1
	i += number
m = len(realdic)
print 'Construct vectors finished.'
print cnt
print m

fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/vectors_'+filename+'.detail', 'w')
for key in realdic:
	fw.write(key)
	for item in realdic[key]:
		fw.write('\t')
		fw.write(str(item)+':'+str(realdic[key][item]))
	fw.write('\n')
fw.close()
print 'Write vectors finished.'

#pearson = np.zeros((m, m))
#jaccard = np.zeros((m, m))
pearson = 0
jaccard = 0
numbers = m * (m - 1) / 2
pointlist = realdic.keys()
pdb = {}
jdb = {}
bins = 10000
wid1 = 1 / bins
wid2 = 2 / bins
end1 = wid1
end2 = -1 + wid2
for i in range(bins):
	pdb[end2] = 0
	jdb[end1] = 0
	end1 += wid1
	end2 += wid2

fw1 = open(prefix+'similarity/'+filename+'_pearson.detail', 'w')
fw2 = open(prefix+'similarity/'+filename+'_jaccard.detail', 'w')
for i in range(m):
	for j in range(i+1, m):
		pij, jij = calcPJ(realdic[pointlist[i]], realdic[pointlist[j]])
		binp = math.ceil(pij * 5000 + 1e-300) / 5000
		binj = math.ceil(jij * 10000 + 1e-300) / 10000
		fw1.write(pointlist[i]+'\t'+pointlist[j]+'\t'+str(pij)+'\n')
		fw2.write(pointlist[i]+'\t'+pointlist[j]+'\t'+str(jij)+'\n')
		pdb[binp] += 1
		jdb[binj] += 1
	print i
fw1.close()
fw2.close()

