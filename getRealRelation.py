import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

single = False

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/all/behavior/'

namelist = os.listdir(path)
realdata = list()
cnt = 0
for name in namelist:
	fr = open(path+name, 'r')
	realdata = fr.readlines()
	relation = {}
	tweets = {}
	fr.close()
	n = len(realdata)
	i = 0
	while i < n:
		temp = realdata[i].split('\t')
		number = int(temp[1]) + 1
		for j in range(i+1, i+number):
			info = realdata[j][:-1].split('\t')
			if not tweets.has_key(info[1]):
				tweets[info[1]] = 1
			else:
				tweets[info[1]] += 1
			if not info[3] == '-1':
				if not relation.has_key(info[4]):
					#if info[4][-1] == '\n':
					#	print info
					relation[info[4]] = {}
					relation[info[4]][info[1]] = 1
				else:
					if not relation[info[4]].has_key(info[1]):
						relation[info[4]][info[1]] = 1
					else:
						relation[info[4]][info[1]] += 1
		i += number
	fw = open('../../cascading_generation_model/all/tweettimes/'+name+'.detail', 'w')
	for key in tweets:
		fw.write(key)
		fw.write('\t')
		fw.write(str(tweets[key]))
		fw.write('\n')
	fw.close()

	fw = open('../../cascading_generation_model/all/relations/'+name+'.detail', 'w')
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
	cnt += 1
	print cnt
print 'Finished.'
