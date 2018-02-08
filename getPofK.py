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

ts = 1321286400 #start timestamps
te = 1322150400 #end timestamps
mid = (ts + te) / 2
te = mid

namelist = os.listdir(path)
points_post = {}
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
	tm = int(temp[i+1].split('\t')[2])
	if tm > mid:
		i += number
		continue
	root = temp[0]
	root_tweet = 0
	casdic = {}
	for j in range(i+1, i+number):
		info = realdata[j][:-1].split('\t')
		if int(info[2]) > mid:
			continue
		authordic[info[0]] = info[1]
		casdic[info[0]] = 0
		if not info[3] == '-1':
			#if not casdic.has_key(info[3]):
			#	casdic[info[3]] = 1
			#else:
			if info[3] == root:
				root_tweet += 1
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
	author = authordic[root]
	if not points_post.has_key(author):
		points_post[author] = {}
	if not points_post[author].has_key(root_tweet):
		points_post[author][root_tweet] = 1
	else:
		points_post[author][root_tweet] += 1
	i += number

fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/PofK_5.detail', 'w')
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
fw.close()

fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/PofK_5_post.detail', 'w')
for key in points_post:
	fw.write(key)
	num = sorted(points_post[key].keys())
	s = sum(points_post[key].values())
	for item in num:
		fw.write('\t')
		fw.write(str(item))
		fw.write(':')
		fw.write(str(points_post[key][item] * 1.0 / s))
	fw.write('\n')
fw.close()
