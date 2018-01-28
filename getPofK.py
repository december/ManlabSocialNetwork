import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

single = False

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268

namelist = os.listdir(path)
points = {}
authordic = {}
realdata = list()
for name in namelist:
	if single and name.startswith(str(filename) + '_'):
		fr = open(path+name, 'r')
		realdata = fr.readlines()
		fr.close()
		break
	if not single:
		fr = open(path+name, 'r')
		realdata.extend(fr.readlines())
		fr.close()

n = len(realdata)
i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	casdic = {}
	for j in range(i+1, i+number):
		info = realdata[j][:-1].split('\t')
		authordic[info[0]] = info[1]
		if not info[3] == '-1':
			if not casdic.has_key(info[3]):
				casdic[info[3]] = 1
			else:
				casdic[info[3]] += 1
	for key in casdic:
		author = authordic[key]
		if not points.has_key(author):
			points[author] = {}
		if not points[author].has_key(casdic[key]):
			points[author][casdic[key]] = 1
		else:
			points[author][casdic[key]] += 1
	i += number

fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/PofK.detail', 'w')
for key in points:
	fw.write(key)
	num = sorted(points[key].keys())
	s = sum(points[key].values())
	for item in num:
		fw.write('\t')
		fw.write(str(item))
		fw.write(':')
		fw.write(str(points[key][item] * 1.0 / s))
	fw.write('\n')
