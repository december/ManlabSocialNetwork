import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

simulation = False
filename = 'Real'
if len(sys.argv) > 1:
	filename = sys.argv[1]
	simulation = True

def WienerIndex(gdic, j, k):
	wi = 0
	if gdic[j][1] > gdic[k][1]:
		while gdic[j][1] > gdic[k][1]:
			j = gdic[j][0]
			wi += 1
		if j == k:
			return wi
	if gdic[j][1] < gdic[k][1]:
		while gdic[j][1] < gdic[k][1]:
			k = gdic[k][0]
			wi += 1
		if j == k:
			return wi
	while j != k:
		j = gdic[j][0]
		k = gdic[k][0]
		wi += 2
	return wi

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268

realdata = list()
real = {} #from cascade id to info list(size, depth, width, wiener index)
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
	number = int(temp[1])
	cascade = temp[0]
	templist = list()
	templist.append(number)
	depdic = {}
	widdic = {}
	graphdic = {}
	keylist = list()
	for j in range(i+1, i+number+1):
		info = realdata[j].split('\t')
		authordic[info[0]] = info[1]
		if info[3] == '-1':
			depdic[info[0]] = 1
		else:
			depdic[info[0]] = depdic[info[3]] + 1
		if widdic.has_key(info[3]):
			widdic[info[3]] += 1
		else:
			widdic[info[3]] = 1
		keylist.append(info[0])
		graphdic[info[0]] = list()
		graphdic[info[0]].append(info[3])
		graphdic[info[0]].append(depdic[info[0]])
	templist.append(max(depdic.values()))
	templist.append(max(widdic.values()))
	wi = 0
	m = len(keylist)
	for j in range(m):
		for k in range(j+1, m):
			wi += WienerIndex(graphdic, keylist[j], keylist[k])
	templist.append(wi)
	real[cascade] = templist
	i += number + 1

fw = open(prefix+'info/'+filename+'.detail', 'w')
for cas in real:
	fw.write(cas)
	for item in real[cas]:
		fw.write('\t')
		fw.write(str(item))
	fw.write('\n')
fw.close()

