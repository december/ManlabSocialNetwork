import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/all/behavior_origin/'
newpath = '../../cascading_generation_model/all/behavior/'

def isRepeat(tid, pr, author):
	pid = pr[tid]
	author1 = author[pr[pr[pid]]]
	author2 = author[pr[pid]]
	author3 = author[pid]
	author4 = author[tid]
	return (author1 == author3 and author2 == author4)

def Connect(info):
	s = info[0]
	m = len(info)
	for i in range(1, m):
		s += '\t'
		s += info[i]
	return s

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
		#newdata.append(realdata[i])
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
				while reflectdic.has_key(info[3]):
					info[3] = reflectdic[info[3]]
					tempdata.pop()
					tempdata.append(Connect(info))
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
		l = len(tempdata)
		#if temp[0] == '418065':
		#	print 'arrive'
		for j in range(l):
			info = tempdata[j].split('\t')
			while reflectdic.has_key(info[3]):
				info[3] = reflectdic[info[3]]
				tempdata[j] = Connect(info)
		newdata.append(temp[0]+'\t'+str(len(tempdata))+'\n')
		newdata.extend(tempdata)
		i += number
	print name + ': ' + str(len(realdata)) + ' to ' + str(len(newdata))
	fw = open(newpath+name, 'w')
	for line in newdata:
		fw.write(line)
	fw.close()
