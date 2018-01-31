import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

single = False

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/all/behavior_old/'

namelist = os.listdir(path)
realdata = list()
count = 0
for name in namelist:
	fr = open(path+name, 'r')
	realdata = fr.readlines()
	relation = {}
	tweets = {}
	fr.close()
	n = len(realdata)
	i = 0
	newdata = list()
	while i < n:
		temp = realdata[i].split('\t')
		tempdata = list()
		tempdata.append(temp[0])
		cnt = 0
		i += 1
		length = len(realdata[i].split('\t'))
		while length != 2:
			tempdata.append(realdata[i])
			i += 1
			cnt += 1
			if i >= n:
				break
			length = len(realdata[i].split('\t'))
		tempdata[0] = tempdata[0] + '\t' + str(cnt) + '\n'
		newdata.extend(tempdata)

	fw = open('../../cascading_generation_model/all/behavior/'+name+'.detail', 'w')
	for line in newdata:
		fw.write(line)
	fw.close()
	count += 1
	print count
print 'Finished.'
