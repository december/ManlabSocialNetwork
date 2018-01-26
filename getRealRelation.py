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
relation = {}
tweets = {}
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
	for j in range(i+1, i+number):
		info = realdata[j].split('\t')
		if not tweets.has_key(info[1]):
			tweets[info[1]] = 1
		else:
			tweets[info[1]] += 1
		if not info[3] == '-1':
			if not relation.has_key(info[4]):
				relation[info[4]] = {}
				relation[info[4]][info[1]] = 1
			else:
				if not relation[info[4]].has_key(info[1]):
					relation[info[4]][info[1]] = 1
				else:
					relation[info[4]][info[1]] += 1
	i += number

fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/tweettimes.detail', 'w')
for key in tweets:
	fw.write(key)
	fw.write('\t')
	fw.write(str(tweets[key]))
	fw.write('\n')
fw.close()

fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/relations.detail', 'w')
for key in relation:
	fw.write(key)
	fw.write('\t')
	fw.write(str(len(relation[key])))
	fw.write('\n')
	for fd in relation[key]:
		fw.write('\t')
		fw.write(fd)
		fw.write('\t')
		fw.write(str(relation[key][fd]))
		fw.write('\n')
fw.close()
