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

participation = list()
par_answer = list()
popularity = list()
pop_answer = list()

n = len(realdata)
i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	info = realdata[i+1].split('\t')
	tm = int(info[2])
	if tm <= mid:
		i += number
		continue
	if number > 10:
		participation.append(info[1]+'\t'+str(number-1)+'\n')
		answer = info[0]
		for j in range(i+1, i+number):
			newinfo = realdata[j].split('\t')
			answer += '\t' + newinfo[1]
		answer += '\n'
		par_answer.append(answer)
	if number > 20:
		pop_answer.append(info[0]+'\t'+str(number-1)+'\n')
		question = info[0]
		for j in range(i+1, i+6):
			newinfo = realdata[j].split('\t')
			question += '\t' + newinfo[1]
		question += '\n'
		popularity.append(question)		
	i += number

fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/Participation.detail', 'w')
for line in participation:
	fw.write(line)
fw.close()
fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/Participation_answer.detail', 'w')
for line in par_answer:
	fw.write(line)
fw.close()
fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/Popularity.detail', 'w')
for line in popularity:
	fw.write(line)
fw.close()
fw = open('../../cascading_generation_model/722911_twolevel_neighbor_cascades/Popularity_answer.detail', 'w')
for line in pop_answer:
	fw.write(line)
fw.close()
