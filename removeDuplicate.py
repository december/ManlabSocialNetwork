import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

prefix = '../../cascading_generation_model/simulation/'
if int(sys.argv[2]) == 0:
	prefix = '../../cascading_generation_model/simulation_notopic/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
newpath = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post_removed/'

def isRepeat(tid, pr, author):
	pid = pr[tid]
	author1 = author[pr[pr[pid]]]
	author2 = author[pr[pid]]
	author3 = author[pid]
	author4 = author[tid]
	return (author1 == author3 and author2 == author4)

namelist = os.listdir(path)
real = {}
sim = {}
realdata = list()
for name in namelist:
	fr = open(path+name, 'r')
	#realdata.extend(fr.readlines())
	realdata = fr.readlines()
	fr.close()
	n = len(realdata)
	i = 0
	newdata = list()
	while i < n:
		newdata.append(realdata[i])
		temp = realdata[i].split('\t')
		number = int(temp[1]) + 1
		tempdata = list()
		depdic = {}
		prdic = {}
		authordic = {}
		reflectdic = {}
		for j in range(i+1, i+number):
			tempdata.append(realdata[j])
			info = realdata[j].split('\t')
			authordic[info[0]] = info[1]
			if info[3] == '-1':
				depdic[info[0]] = 0
			else:
				prdic[info[0]] = info[3]
				tempdep = depdic[info[3]] + 1
				depdic[info[0]] = tempdep
				if not reflectdic.has_key(info[3]) and tempdep >= 3:
					if isRepeat(info[0], prdic, authordic):
						tempdata.pop()
						l = len(tempdata) - 1
						while l >= 0:
							if tempdata[l].split('\t')[0] == info[3]:
								del tempdata[l]
								break
							l -= 1
						reflectdic[info[3]] = prdic[prdic[info[3]]]
						reflectdic[info[0]] = prdic[prdic[info[0]]]
		newdata.extend(tempdata)
		i += number
	print name + ': ' + str(len(realdata)) + ' to ' + str(len(newdata))
	fw = open(newpath+name, 'w')
	for line in newdata:
		fw.write(line)
	fw.close()
