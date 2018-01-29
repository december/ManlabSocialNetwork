import sys     
import scipy as sp
import numpy as np
import numpy.random
import random

def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()
import signal
signal.signal(signal.SIGINT, debug_signal_handler)

single = True
filename = int(sys.argv[1])
ts = 0
te = float(sys.argv[2]) * 86400
sims = int(sys.argv[3])
if filename < 0:
	single = False

users = 7268
vnum = 7268
enum = 7268
edgemap = {}
uid = list() #from user index in this ego network to user id
iddic = {} #from user id to user index in this ego network
tweetdic = {} #from tweet id to the user index of its author
number = 0 #total number of tweeters
pkdic = {}

def GetOffspring(key):
	p = numpy.random.rand()
	for times in pkdic[key]:
		p -= pkdic[key][times]
		if p <= 0:
			return times
	print 'Empty!'
	return pkdic[key].keys()[-1]

def GetLog(r, p, u, c): #root_tweet, parent_tweet, parent_user, parent_time, tau, cascade log, depth
	global number
	#if d >= 100:
	#	return c
	if not edgemap.has_key(u):
		return c
	m = GetOffspring(u)
	offspring = random.sample(edgemap[u], m)
	for f in offspring:
		current = number
		tweetdic[current] = f
		number += 1
		temp = list()
		temp.append(current)
		temp.append(f)
		temp.append(0)
		temp.append(p)
		temp.append(u)
		c.append(temp)
		c = GetLog(r, current, f, c)
	return c

prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'

postdic = {}
fr = open(prefix+'posttimes'+suffix, 'r')
timelist = fr.readlines()
postlist = list()
for i in range(users):
	temp = timelist[i].split('\t')
	postdic[temp[0]] = int(temp[1])
fr.close()

if single:
	prefix = prefix + 'single_user_parameter/'
	suffix = '_' + str(filename) + suffix
#if int(sys.argv[4]) == 0:
#	suffix = '_notopic.detail'

fr = open(prefix+'relations'+suffix, 'r')
relation = fr.readlines()
fr.close()
n = len(relation)
i = 0
while i < n:
	temp = relation[i].split('\t')
	number = int(temp[1]) + 1
	if not edgemap.has_key(temp[0]):
		edgemap[temp[0]] = list()
	for j in range(i+1, i+number):
		info = relation[j].split('\t')
		edgemap[temp[0]].append(info[1])
	i += number


fr = open(prefix+'PofK'+suffix, 'r')
pk = fr.readlines()
fr.close()
for line in pk:
	temp = line.split('\t')
	n = len(temp)
	pkdic[temp[0]] = {}
	for j in range(1, n):
		info = temp[j].split(':')
		pkdic[temp[0]][int(info[0])] = float(info[1])

#print iddic

#for key in lbddic:
#	lbd[iddic[key]] = lbddic[key]
#x -= 1.95

print 'Finished reading..'
prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
if single:
	prefix += str(filename) + '/'

for j in range(sims):
	casnum = 0
	number = 0
	behavior = list()
	print 'Generation ' + str(j+1) + ' begins...'
	for key in postdic:
		print key
		for i in range(postdic[key]):
			#tweetdic[number] = newi
			root = number
			number += 1
			cascade = list()
			temp = list()
			temp.append(root)
			temp.append(key)
			temp.append(ts)
			temp.append(-1)
			temp.append(-1)
			cascade.append(temp)
			cascade = GetLog(root, root, key, cascade)
			cascade = sorted(cascade, key=lambda c:c[2])
			size = len(cascade)
			temp = list()
			temp.append(root)
			temp.append(size)
			behavior.append(temp)
			behavior.extend(cascade)
			casnum += 1
	print casnum
	print number
	fw = open(prefix+str(j)+suffix, 'w')
	for item in behavior:
		fw.write(str(item[0]))
		n = len(item)
		for k in range(1, n):
			fw.write('\t')
			fw.write(str(item[k]))
		fw.write('\n')
	fw.close()

print 'Finished generation task...'
