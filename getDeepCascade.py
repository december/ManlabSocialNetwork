import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

filename = int(sys.argv[1])
if filename < 0:
	single = False

prefix = '../../cascading_generation_model/simulation/'
if int(sys.argv[2]) == 0:
	prefix = '../../cascading_generation_model/simulation_notopic/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268

namelist = os.listdir(path)
real = {}
sim = {}
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
deep = list()
while i < n:
	temp = realdata[i].split('\t')
	tempdeep = list()
	tempdeep.append(realdata[i])
	number = int(temp[1]) + 1
	depdic = {}
	for j in range(i+1, i+number):
		tempdeep.append(realdata[j])
		info = realdata[j].split('\t')
		if info[3] == '-1':
			depdic[info[0]] = 0
		else:
			depdic[info[0]] = depdic[info[3]] + 1
	dep = max(depdic.values())
	if dep > 10:
		deep.extend(tempdeep)
	i += number

fw = open(prefix+'DepthDistribution/DeepCascades.detail', 'w')
for line in deep:
	fw.write(line)
fw.close()

