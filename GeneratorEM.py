import sys
import scipy as sp
import numpy as np
import numpy.random

single = True
filename = int(sys.argv[1])
ts = 0
te = int(float(sys.argv[2])) * 86400
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

if len(sys.argv) > 4:
	prefix += sys.argv[4] + '/'

fr = open(prefix+'gamma'+suffix, 'r')
omglist = fr.readlines()
vnum = len(omglist)

lbd = np.zeros(vnum) #parameter lambda which have calculated before
gamma = np.zeros(vnum) #parameter omega

for i in range(vnum):
	temp = omglist[i].split('\t')
	uid.append(temp[0])
	iddic[int(temp[0])] = i
	gamma[i] = float(temp[1])
fr.close()
#print iddic

#for key in lbddic:
#	lbd[iddic[key]] = lbddic[key]

fr = open(prefix+'beta'+suffix, 'r')
pilist = fr.readlines()
enum = len(pilist)

beta = np.zeros(enum) #parameter pi (based on edges), row is sender while col is receiver

for i in range(enum):
	temp = pilist[i].split('\t')
	row = iddic[int(temp[0])]
	col = iddic[int(temp[1])]
	if not edgemap.has_key(row):
		edgemap[row] = {}
	edgemap[row][col] = i
	beta[i] = float(temp[2])
fr.close()

print 'Finished reading..'
prefix = '../../cascading_generation_model/simulation_notopic/'
suffix = '.detail'
if single:
	prefix += str(filename) + '/'

def GetLog(r, p, u, c): #root_tweet, parent_tweet, parent_user, parent_time, tau, cascade log, depth
	global number
	if not edgemap.has_key(u):
		return c
	if np.random.rand() <= gamma[u]:
		return c
	for f in edgemap[u]:
		if np.random.rand() <= beta[edgemap[u][f]]:
			current = number
			tweetdic[current] = f
			number += 1
			temp = list()
			temp.append(current)
			temp.append(uid[f])
			temp.append(0)
			temp.append(p)
			temp.append(uid[u])
			c.append(temp)
			c = GetLog(r, current, f, c)
	return c

for j in range(sims):
	number = 0
	behavior = list()
	print 'Generation ' + str(j+1) + ' begins...'
	casnum = 0
	totalnum = 0
	for key in postdic:
		print key
		newi = iddic[int(key)]
		for i in range(postdic[key]):
			casnum += 1
			tweetdic[number] = newi
			root = number
			number += 1
			cascade = list()
			temp = list()
			temp.append(root)
			temp.append(uid[newi])
			temp.append(ts)
			temp.append(-1)
			temp.append(-1)
			cascade.append(temp)
			#tau = GetTau(phi1, phi2, phi3, phi4, phi5, i)
			cascade = GetLog(root, root, newi, cascade)
			cascade = sorted(cascade, key=lambda c:c[2])
			size = len(cascade)
			temp = list()
			temp.append(root)
			temp.append(size)
			behavior.append(temp)
			behavior.extend(cascade)
			totalnum += size
	print casnum
	print totalnum
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
